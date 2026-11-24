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
"""Small-footprint CuTe DSL o_proj for weights staged in L2 (SM100/SM103).

``out[r, n] = bf16(sum_k v[r, k] * w_o[n, k])`` with FP32 accumulation, at
most 16 live rows ``r``, like ``cute_oproj.stream_oproj`` but on ``grid``
CTAs (default 96, the footprint of the native o_proj GEMM) instead of one per
SM, so a register-heavy PDL successor (the post_attention_hop needs a whole SM per CTA)
can become resident on the remaining SMs while this kernel runs.

* CTA ``b`` owns ``rows = N / grid`` contiguous ``w_o`` rows, processed as
  16-row tiles through a ring of ``slots`` row-padded shared-memory tiles
  (pitch ``K + 8`` elements, bank-conflict-free operand loads). Each tile
  completes on its own mbarrier; a slot is refilled with the tile
  ``slots`` positions later once all warps have consumed it.
* The first ``early`` tiles are requested before ``griddepcontrol.wait``
  (only immutable weights are read before the wait); everything else, and
  every read of ``v``, follows it.
* Eight warps split K into 256-wide ranges, keep their ``v`` fragments in
  registers and run ``mma.sync.m16n8k16`` (BF16 in, FP32 accumulate); the
  per-warp partials are added in fixed warp order (deterministic) and rounded
  once to BF16. Rows ``[num_rows, out_rows)`` are written as zeros.
* The PDL successor is released once this CTA's MMAs are issued after the
  wait (``trigger``); it still waits for the whole grid before reading.
* ``publish`` (not supported; ``ring_oproj`` refuses it): the epilogue would
  additionally write the live rows of the partial into slot ``[row][rank]``
  of the current Lamport buffer of the post_attention_hop workspace. The into-hop entry polls, and
  later resets, every row of the hop including padding, so a producer that
  publishes only the live rows leaves a padded call spinning or leaves stale
  words behind; attention_output_into_hop publishes through ``into_hop``, which writes every
  hop row.
"""

import functools
from typing import Optional

import torch

_THREADS = 256
_WARPS = _THREADS // 32
_ROW_PAD = 8
_TILE_M = 16
_HIDDEN = 6144
_NUM_RANKS = 8
_LAMPORT_BUFFER_WORDS = 8 * _NUM_RANKS * _HIDDEN * 2 // 4


def plan(n_dim: int, k_dim: int, grid: int, slots: int) -> dict:
    rows = n_dim // grid
    pitch = k_dim + _ROW_PAD
    return {
        "rows": rows,
        "tiles": rows // _TILE_M,
        "pitch": pitch,
        "smem_bytes": slots * _TILE_M * pitch * 2,
        "k_per_warp": k_dim // _WARPS,
    }


def check_shapes(n_dim: int, k_dim: int, grid: int, slots: int) -> Optional[str]:
    """Return why the kernel cannot take this ``[n_dim, k_dim]`` weight."""
    if k_dim % (_WARPS * 16) or (k_dim * 2) % 16:
        return "K is not a multiple of 128"
    if n_dim % (grid * _TILE_M):
        return "N is not a multiple of grid * 16"
    p = plan(n_dim, k_dim, grid, slots)
    if p["smem_bytes"] > 210 * 1024:
        return "ring does not fit in shared memory"
    if p["smem_bytes"] < _WARPS * p["tiles"] * 2 * 128 * 4:
        return "ring smaller than the reduction buffer"
    return None


@functools.lru_cache(maxsize=None)
def _build(
    n_dim: int,
    k_dim: int,
    grid: int,
    slots: int,
    early: int,
    nt: int,
    use_pdl: bool,
    trigger: bool,
    publish_rank: int,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cute import runtime as cute_runtime
    from cutlass.cutlass_dsl import T, dsl_user_op
    from cutlass.utils.smem_allocator import SmemAllocator

    from tensorrt_llm._torch.kernels.blackwell.cute_ptx_utils import mma_sync

    geo = plan(n_dim, k_dim, grid, slots)
    rows = geo["rows"]
    tiles = geo["tiles"]
    pitch = geo["pitch"]
    half_pitch = pitch // 2
    k_per_warp = geo["k_per_warp"]
    steps = k_per_warp // 16
    row_bytes = k_dim * 2
    red_stride = 128
    publish = publish_rank >= 0
    slots_used = min(slots, tiles)
    early = min(early, slots_used)

    def _asm(res, operands, text, constraints, loc, ip):
        return llvm.inline_asm(
            res, operands, text, constraints, has_side_effects=True, asm_dialect=0, loc=loc, ip=ip
        )

    @dsl_user_op
    def bulk_g2s(dst_smem, src_gmem, num_bytes, mbar, *, loc=None, ip=None):
        _asm(
            None,
            [
                dst_smem.toint(loc=loc, ip=ip).ir_value(),
                src_gmem.toint(loc=loc, ip=ip).ir_value(),
                Int32(num_bytes).ir_value(loc=loc, ip=ip),
                mbar.toint(loc=loc, ip=ip).ir_value(),
            ],
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [$0], [$1], $2, [$3];",
            "r,l,r,r",
            loc,
            ip,
        )

    @dsl_user_op
    def grid_dependency_wait(*, loc=None, ip=None):
        _asm(None, [], "griddepcontrol.wait;", "", loc, ip)

    @dsl_user_op
    def grid_launch_dependents(*, loc=None, ip=None):
        _asm(None, [], "griddepcontrol.launch_dependents;", "", loc, ip)

    @dsl_user_op
    def ld_volatile_u32(addr, *, loc=None, ip=None):
        return Int32(
            _asm(
                T.i32(),
                [Int64(addr).ir_value(loc=loc, ip=ip)],
                "ld.volatile.global.u32 $0, [$1];",
                "=r,l",
                loc,
                ip,
            )
        )

    @dsl_user_op
    def st_global_b32(addr, val, *, loc=None, ip=None):
        _asm(
            None,
            [Int64(addr).ir_value(loc=loc, ip=ip), Int32(val).ir_value(loc=loc, ip=ip)],
            "st.global.b32 [$0], $1;",
            "l,r",
            loc,
            ip,
        )

    class RingOproj:
        @cute.kernel
        def kernel(
            self,
            v: cute.Tensor,
            w: cute.Tensor,
            out: cute.Tensor,
            mc: cute.Tensor,
            flags: cute.Tensor,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            warp = tidx // Int32(32)
            lane = tidx % Int32(32)
            grp = lane // Int32(4)
            quad = lane % Int32(4)

            num_rows = Int32(v.shape[0])
            out_rows = Int32(out.shape[0])
            n0 = bidx * Int32(rows)

            smem = SmemAllocator()
            slab = smem.allocate_tensor(
                BFloat16, cute.make_layout((slots_used * _TILE_M * pitch,)), byte_alignment=128
            )
            mbar = smem.allocate_tensor(
                cutlass.Int64, cute.make_layout((slots_used,)), byte_alignment=8
            )
            slab32 = cute.recast_tensor(slab, Int32)
            slab_f32 = cute.recast_tensor(slab, Float32)
            v32 = cute.recast_tensor(v, Int32)

            if tidx == Int32(0):
                for s in cutlass.range_constexpr(slots_used):
                    cute.arch.mbarrier_init(mbar.iterator + s, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            # Tile t goes to slot t % slots: lane 0 of warp 0 arms the slot's
            # mbarrier, lanes 0..15 copy one row each.
            if warp == Int32(0):
                for t in cutlass.range_constexpr(early):
                    if lane == Int32(0):
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar.iterator + t, Int32(_TILE_M * row_bytes)
                        )
                    cute.arch.sync_warp()
                    if lane < Int32(_TILE_M):
                        bulk_g2s(
                            slab.iterator + (Int32(t * _TILE_M) + lane) * Int32(pitch),
                            w.iterator + (n0 + Int32(t * _TILE_M) + lane) * Int32(k_dim),
                            row_bytes,
                            mbar.iterator + t,
                        )

            if cutlass.const_expr(use_pdl):
                grid_dependency_wait()

            if warp == Int32(0):
                for t in cutlass.range_constexpr(early, slots_used):
                    if lane == Int32(0):
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbar.iterator + t, Int32(_TILE_M * row_bytes)
                        )
                    cute.arch.sync_warp()
                    if lane < Int32(_TILE_M):
                        bulk_g2s(
                            slab.iterator + (Int32(t * _TILE_M) + lane) * Int32(pitch),
                            w.iterator + (n0 + Int32(t * _TILE_M) + lane) * Int32(k_dim),
                            row_bytes,
                            mbar.iterator + t,
                        )

            # v fragments of this warp's K range (B operand, [n][k] storage).
            vb = cute.make_rmem_tensor((4, nt, steps), BFloat16)
            vb32 = cute.recast_tensor(vb, Int32)
            k_word0 = warp * Int32(k_per_warp // 2) + quad
            for s in cutlass.range_constexpr(steps):
                for j in cutlass.range_constexpr(nt):
                    r = Int32(8 * j) + grp
                    lo = Int32(0)
                    hi = Int32(0)
                    if r < num_rows:
                        lo = v32[r, k_word0 + Int32(8 * s)]
                        hi = v32[r, k_word0 + Int32(8 * s + 4)]
                    vb32[0, j, s] = lo
                    vb32[1, j, s] = hi

            if cutlass.const_expr(trigger):
                # After the wait: every grid before this one has completed.
                grid_launch_dependents()

            acc = cute.make_rmem_tensor((4, nt, tiles), Float32)
            acc.fill(0.0)
            a = cute.make_rmem_tensor((8,), BFloat16)
            a32 = cute.recast_tensor(a, Int32)
            for t in cutlass.range_constexpr(tiles):
                s = t % slots_used
                cute.arch.mbarrier_wait(mbar.iterator + s, (t // slots_used) % 2)
                base_lo = (Int32(s * _TILE_M) + grp) * Int32(half_pitch) + (k_word0)
                base_hi = base_lo + Int32(8 * half_pitch)
                for st in cutlass.range_constexpr(steps):
                    a32[0] = slab32[base_lo + Int32(8 * st)]
                    a32[1] = slab32[base_hi + Int32(8 * st)]
                    a32[2] = slab32[base_lo + Int32(8 * st + 4)]
                    a32[3] = slab32[base_hi + Int32(8 * st + 4)]
                    for j in cutlass.range_constexpr(nt):
                        acc[None, j, t] = mma_sync(a, vb[None, j, st], acc[None, j, t])
                if cutlass.const_expr(t + slots_used < tiles):
                    # Every warp has read slot s: refill it with tile
                    # t + slots.
                    cute.arch.sync_threads()
                    if warp == Int32(0):
                        if lane == Int32(0):
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar.iterator + s, Int32(_TILE_M * row_bytes)
                            )
                        cute.arch.sync_warp()
                        if lane < Int32(_TILE_M):
                            bulk_g2s(
                                slab.iterator + (Int32(s * _TILE_M) + lane) * Int32(pitch),
                                w.iterator
                                + (n0 + Int32((t + slots_used) * _TILE_M) + lane) * Int32(k_dim),
                                row_bytes,
                                mbar.iterator + s,
                            )

            # Fixed-order cross-warp reduction through the (now idle) ring.
            cute.arch.sync_threads()
            for t in cutlass.range_constexpr(tiles):
                for j in cutlass.range_constexpr(nt):
                    base = ((warp * Int32(tiles) + Int32(t)) * Int32(nt) + Int32(j)) * Int32(
                        red_stride
                    ) + lane * Int32(4)
                    for e in cutlass.range_constexpr(4):
                        slab_f32[base + Int32(e)] = acc[e, j, t]
            cute.arch.sync_threads()

            out_limit = out_rows
            if out_limit > Int32(8 * nt):
                out_limit = Int32(8 * nt)
            buf_word0 = Int32(0)
            if cutlass.const_expr(publish):
                buf_word0 = ld_volatile_u32(flags.iterator.toint()) * Int32(_LAMPORT_BUFFER_WORDS)
            mc_base = mc.iterator.toint()
            # Pairs of adjacent output columns (m, m + 1) of one row: lane
            # groups 2g and 2g + 1 of the MMA C fragment.
            pk = cute.make_rmem_tensor((2,), BFloat16)
            pk32 = cute.recast_tensor(pk, Int32)
            total_pairs = tiles * nt * 64
            p = tidx
            while p < Int32(total_pairs):
                t_i = p // Int32(nt * 64)
                j_i = (p // Int32(64)) % Int32(nt)
                q = p % Int32(64)
                gh = q // Int32(16)
                quad_i = (q // Int32(4)) % Int32(4)
                e_i = q % Int32(4)
                idx0 = (
                    (t_i * Int32(nt) + j_i) * Int32(red_stride)
                    + (gh * Int32(8) + quad_i) * Int32(4)
                    + e_i
                )
                idx1 = idx0 + Int32(16)
                s0 = Float32(0.0)
                s1 = Float32(0.0)
                for wi in cutlass.range_constexpr(_WARPS):
                    s0 = s0 + slab_f32[Int32(wi * tiles * nt * red_stride) + idx0]
                    s1 = s1 + slab_f32[Int32(wi * tiles * nt * red_stride) + idx1]
                m0 = t_i * Int32(_TILE_M) + gh * Int32(2) + Int32(8) * (e_i // Int32(2))
                r = j_i * Int32(8) + Int32(2) * quad_i + e_i % Int32(2)
                if r < out_limit:
                    pk[0] = s0.to(BFloat16)
                    pk[1] = s1.to(BFloat16)
                    out[r, n0 + m0] = pk[0]
                    out[r, n0 + m0 + Int32(1)] = pk[1]
                    if cutlass.const_expr(publish):
                        if r < num_rows:
                            word = pk32[0]
                            lo = word & Int32(0xFFFF)
                            hi = (word >> Int32(16)) & Int32(0xFFFF)
                            if lo == Int32(0x8000):
                                lo = Int32(0)
                            if hi == Int32(0x8000):
                                hi = Int32(0)
                            word = lo | (hi << Int32(16))
                            widx = buf_word0 + (
                                (r * Int32(_NUM_RANKS) + Int32(publish_rank)) * Int32(_HIDDEN)
                                + n0
                                + m0
                            ) // Int32(2)
                            st_global_b32(mc_base + Int64(widx) * Int64(4), word)
                p = p + Int32(_THREADS)

        @cute.jit
        def __call__(
            self,
            v: cute.Tensor,
            w: cute.Tensor,
            out: cute.Tensor,
            mc: cute.Tensor,
            flags: cute.Tensor,
            stream,
        ):
            self.kernel(v, w, out, mc, flags).launch(
                grid=(grid, 1, 1),
                block=(_THREADS, 1, 1),
                stream=stream,
                use_pdl=use_pdl,
                min_blocks_per_mp=1,
            )

    rows_sym, out_sym = cute.sym_int(), cute.sym_int()
    mc_sym = cute.sym_int()

    def fake(dtype, shape):
        return cute_runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=tuple(range(len(shape) - 1, -1, -1)), assumed_align=16
        )

    return cute.compile(
        RingOproj(),
        fake(BFloat16, (rows_sym, k_dim)),
        fake(BFloat16, (n_dim, k_dim)),
        fake(BFloat16, (out_sym, n_dim)),
        fake(Int32, (mc_sym,)),
        fake(Int32, (4,)),
        stream=cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


_DUMMY = {}


def _dummy(device: torch.device) -> torch.Tensor:
    key = device.index or 0
    if key not in _DUMMY:
        _DUMMY[key] = torch.zeros(4, dtype=torch.int32, device=device)
    return _DUMMY[key]


def ring_oproj(
    v: torch.Tensor,
    w_o: torch.Tensor,
    out: torch.Tensor,
    use_pdl: bool,
    grid: int = 96,
    slots: int = 3,
    early: int = 3,
    trigger: bool = True,
    mc_words: Optional[torch.Tensor] = None,
    flags: Optional[torch.Tensor] = None,
    publish_rank: int = -1,
) -> None:
    """Launch the small-footprint o_proj on the current stream.

    Args:
        v: ``[num_rows, K]`` contiguous BF16, ``num_rows <= 16``.
        w_o: ``[N, K]`` contiguous BF16, 16-byte aligned.
        out: ``[out_rows, N]`` contiguous BF16; rows ``[0, min(out_rows, 16))``
            are written (rows past ``num_rows`` as zeros).
        use_pdl: launch as a PDL dependent of the previous kernel.
        grid: CTAs; ``N`` must be a multiple of ``16 * grid``.
        slots: 16-row shared-memory tiles per CTA.
        early: tiles requested before ``griddepcontrol.wait``.
        trigger: release the PDL successor after the wait.
        mc_words / flags / publish_rank: unsupported publish target; only
            ``publish_rank < 0`` (no publishing) is accepted.

    Raises:
        ValueError: ``publish_rank >= 0`` (see the module docstring).
    """
    n_dim, k_dim = w_o.shape
    nt = 1 if out.shape[0] <= 8 else 2
    if publish_rank >= 0:
        raise ValueError(
            "ring_oproj publishing is withdrawn: it publishes only the live rows, "
            "the attention_output_into_hop hop needs every hop row; publish with into_hop"
        )
    del mc_words, flags
    mc_words = flags = _dummy(w_o.device)
    compiled = _build(n_dim, k_dim, grid, slots, early, nt, use_pdl, trigger, publish_rank)
    compiled(v, w_o, out, mc_words, flags)


__all__ = ["check_shapes", "plan", "ring_oproj"]
