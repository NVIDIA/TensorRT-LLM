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
"""CuTe DSL streaming o_proj kernel for the attention_output_tail (SM100/SM103).

``out[r, n] = bf16(sum_k v[r, k] * w_o[n, k])`` with FP32 accumulation, for at
most 16 live rows ``r``. The kernel is built to start streaming ``w_o`` before
its input ``v`` exists:

* One persistent CTA per SM owns a contiguous block of 40-41 ``w_o`` rows.
  At entry, warp 0 issues a 1-D ``cp.async.bulk`` global→shared copy per row
  (4 KiB each) into a row-padded shared-memory slab, completing on one
  mbarrier per 16-row tile.
* All threads then execute ``griddepcontrol.wait``. When the kernel is a PDL
  dependent of the W_UV kernel, this is where it waits for ``v``; the first
  ``early_rows`` rows per CTA are already in flight and the rest are issued
  after the wait, so the early stream does not queue ~24 MiB of copies in
  front of the W_UV kernel's own memory traffic. Without PDL every row is
  issued at entry.
* Eight warps split K into 256-wide ranges. Each warp keeps its ``v``
  fragments in registers and runs ``mma.sync.m16n8k16`` (BF16 in, FP32
  accumulate) over each 16-row tile as soon as that tile's mbarrier flips.
* The per-warp partial sums are added in fixed warp order through shared
  memory (reusing the slab), so the result is deterministic, and rounded once
  to BF16. Rows ``[num_rows, out_rows)`` are written as zeros.

The 16-byte row padding (pitch ``K + 8`` elements) makes the MMA operand loads
bank-conflict free.
"""

import functools
import math
from typing import Optional

import torch

_THREADS = 256
_WARPS = _THREADS // 32
_ROW_PAD = 8  # BF16 elements of padding per shared-memory row (16 bytes)
_TILE_M = 16


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def plan(n_dim: int, k_dim: int, grid: int) -> dict:
    """Compile-time geometry of the kernel for one weight shape."""
    rows_max = math.ceil(n_dim / grid)
    pitch = k_dim + _ROW_PAD
    return {
        "rows_max": rows_max,
        "num_tiles": math.ceil(rows_max / _TILE_M),
        "pitch": pitch,
        "smem_bytes": rows_max * pitch * 2,
        "k_per_warp": k_dim // _WARPS,
    }


def check_shapes(n_dim: int, k_dim: int, grid: int) -> Optional[str]:
    """Return why the kernel cannot take this ``[n_dim, k_dim]`` weight."""
    if k_dim % (_WARPS * 16):
        return "K is not a multiple of 128"
    p = plan(n_dim, k_dim, grid)
    red_bytes = _WARPS * p["num_tiles"] * 2 * 32 * 4 * 4
    if p["smem_bytes"] < red_bytes:
        return "slab smaller than the reduction buffer"
    # 227 KB of opt-in shared memory per CTA on SM100/SM103, minus the
    # tile mbarriers and alignment.
    if p["smem_bytes"] > 226 * 1024:
        return "weight slab does not fit in shared memory"
    if (k_dim * 2) % 16:
        return "row bytes not a 16-byte multiple"
    return None


def _build(
    n_dim: int,
    k_dim: int,
    grid: int,
    n_tiles_r: int,
    use_pdl: bool,
    early_rows: int,
    trigger_dependents: bool,
    trigger_late: bool,
):
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32
    from cutlass._mlir.dialects import llvm
    from cutlass.cute import runtime as cute_runtime
    from cutlass.cutlass_dsl import dsl_user_op
    from cutlass.utils.smem_allocator import SmemAllocator

    from tensorrt_llm._torch.kernels.blackwell.cute_ptx_utils import mma_sync

    geo = plan(n_dim, k_dim, grid)
    rows_max = geo["rows_max"]
    num_tiles = geo["num_tiles"]
    pitch = geo["pitch"]
    half_pitch = pitch // 2
    k_per_warp = geo["k_per_warp"]
    steps = k_per_warp // 16
    row_bytes = k_dim * 2
    nt = n_tiles_r
    red_stride = 128  # 32 lanes x 4 accumulators

    @dsl_user_op
    def bulk_g2s(dst_smem, src_gmem, num_bytes, mbar, *, loc=None, ip=None):
        llvm.inline_asm(
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
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def grid_launch_dependents(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            "griddepcontrol.launch_dependents;",
            "",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def grid_dependency_wait(*, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [],
            "griddepcontrol.wait;",
            "",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )

    class StreamOproj:
        @cute.kernel
        def kernel(self, v: cute.Tensor, w: cute.Tensor, out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            warp = tidx // Int32(32)
            lane = tidx % Int32(32)
            grp = lane // Int32(4)
            quad = lane % Int32(4)

            num_rows = Int32(v.shape[0])
            out_rows = Int32(out.shape[0])
            n0 = (bidx * Int32(n_dim)) // Int32(grid)
            n1 = ((bidx + Int32(1)) * Int32(n_dim)) // Int32(grid)
            nrows = n1 - n0

            smem = SmemAllocator()
            slab = smem.allocate_tensor(
                BFloat16, cute.make_layout((rows_max * pitch,)), byte_alignment=128
            )
            mbar = smem.allocate_tensor(
                cutlass.Int64, cute.make_layout((num_tiles,)), byte_alignment=8
            )
            slab32 = cute.recast_tensor(slab, Int32)
            slab_f32 = cute.recast_tensor(slab, Float32)
            v32 = cute.recast_tensor(v, Int32)

            if tidx == Int32(0):
                for t in cutlass.range_constexpr(num_tiles):
                    cute.arch.mbarrier_init(mbar.iterator + t, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

            # Stream this CTA's rows of w into shared memory before waiting
            # for the producer of v: lane 0 arms the tile mbarriers, then
            # warp 0 issues one 1-D bulk copy per row.
            if tidx == Int32(0):
                for t in cutlass.range_constexpr(num_tiles):
                    tile_rows = nrows - Int32(t * _TILE_M)
                    if tile_rows > Int32(_TILE_M):
                        tile_rows = Int32(_TILE_M)
                    if tile_rows < Int32(0):
                        tile_rows = Int32(0)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar.iterator + t, tile_rows * Int32(row_bytes)
                    )
            cute.arch.sync_warp()
            # Rows [0, early_rows) go out before the wait on v's producer and
            # the rest after it: a bounded early stream overlaps the producer
            # without queueing ~24 MiB of copies in front of its traffic.
            if warp == Int32(0):
                for q in cutlass.range_constexpr((rows_max + 31) // 32):
                    row = lane + Int32(32 * q)
                    if row < nrows:
                        if row < Int32(early_rows):
                            bulk_g2s(
                                slab.iterator + row * Int32(pitch),
                                w.iterator + (n0 + row) * Int32(k_dim),
                                row_bytes,
                                mbar.iterator + row // Int32(_TILE_M),
                            )

            if cutlass.const_expr(use_pdl):
                grid_dependency_wait()

            if warp == Int32(0):
                for q in cutlass.range_constexpr((rows_max + 31) // 32):
                    row = lane + Int32(32 * q)
                    if row < nrows:
                        if row >= Int32(early_rows):
                            bulk_g2s(
                                slab.iterator + row * Int32(pitch),
                                w.iterator + (n0 + row) * Int32(k_dim),
                                row_bytes,
                                mbar.iterator + row // Int32(_TILE_M),
                            )
            if cutlass.const_expr(trigger_dependents and not trigger_late):
                # Dependents still wait for this whole grid (and its memory)
                # before they read anything it produces.
                grid_launch_dependents()

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

            acc = cute.make_rmem_tensor((4, nt, num_tiles), Float32)
            acc.fill(0.0)
            a = cute.make_rmem_tensor((8,), BFloat16)
            a32 = cute.recast_tensor(a, Int32)
            last = nrows - Int32(1)
            for t in cutlass.range_constexpr(num_tiles):
                cute.arch.mbarrier_wait(mbar.iterator + t, 0)
                row_lo = Int32(t * _TILE_M) + grp
                row_hi = row_lo + Int32(8)
                if row_lo > last:
                    row_lo = last
                if row_hi > last:
                    row_hi = last
                base_lo = row_lo * Int32(half_pitch) + k_word0
                base_hi = row_hi * Int32(half_pitch) + k_word0
                for s in cutlass.range_constexpr(steps):
                    a32[0] = slab32[base_lo + Int32(8 * s)]
                    a32[1] = slab32[base_hi + Int32(8 * s)]
                    a32[2] = slab32[base_lo + Int32(8 * s + 4)]
                    a32[3] = slab32[base_hi + Int32(8 * s + 4)]
                    for j in cutlass.range_constexpr(nt):
                        acc[None, j, t] = mma_sync(a, vb[None, j, s], acc[None, j, t])

            if cutlass.const_expr(trigger_dependents and trigger_late):
                # Late trigger: only once this CTA's MMAs are done.
                grid_launch_dependents()
            # Fixed-order cross-warp reduction through the (now idle) slab.
            cute.arch.sync_threads()
            for t in cutlass.range_constexpr(num_tiles):
                for j in cutlass.range_constexpr(nt):
                    base = ((warp * Int32(num_tiles) + Int32(t)) * Int32(nt) + Int32(j)) * Int32(
                        red_stride
                    ) + lane * Int32(4)
                    for e in cutlass.range_constexpr(4):
                        slab_f32[base + Int32(e)] = acc[e, j, t]
            cute.arch.sync_threads()

            total = num_tiles * nt * red_stride
            out_limit = out_rows
            if out_limit > Int32(8 * nt):
                out_limit = Int32(8 * nt)
            idx = tidx
            while idx < Int32(total):
                t_i = idx // Int32(nt * red_stride)
                j_i = (idx // Int32(red_stride)) % Int32(nt)
                l_i = (idx // Int32(4)) % Int32(32)
                e_i = idx % Int32(4)
                acc_sum = Float32(0.0)
                for wi in cutlass.range_constexpr(_WARPS):
                    acc_sum = acc_sum + slab_f32[Int32(wi * num_tiles * nt * red_stride) + idx]
                m = t_i * Int32(_TILE_M) + l_i // Int32(4) + Int32(8) * (e_i // Int32(2))
                r = j_i * Int32(8) + Int32(2) * (l_i % Int32(4)) + e_i % Int32(2)
                if m < nrows:
                    if r < out_limit:
                        out[r, n0 + m] = acc_sum.to(BFloat16)
                idx = idx + Int32(_THREADS)

        @cute.jit
        def __call__(self, v: cute.Tensor, w: cute.Tensor, out: cute.Tensor, stream):
            self.kernel(v, w, out).launch(
                grid=(grid, 1, 1),
                block=(_THREADS, 1, 1),
                stream=stream,
                use_pdl=use_pdl,
                min_blocks_per_mp=1,
            )

    rows_sym, out_sym = cute.sym_int(), cute.sym_int()
    v_fake = cute_runtime.make_fake_compact_tensor(
        BFloat16, (rows_sym, k_dim), stride_order=(1, 0), assumed_align=16
    )
    w_fake = cute_runtime.make_fake_compact_tensor(
        BFloat16, (n_dim, k_dim), stride_order=(1, 0), assumed_align=16
    )
    out_fake = cute_runtime.make_fake_compact_tensor(
        BFloat16, (out_sym, n_dim), stride_order=(1, 0), assumed_align=16
    )
    stream = cute_runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        StreamOproj(), v_fake, w_fake, out_fake, stream=stream, options="--enable-tvm-ffi"
    )


_COMPILED = {}


DEFAULT_EARLY_ROWS = 16


def stream_oproj(
    v: torch.Tensor,
    w_o: torch.Tensor,
    out: torch.Tensor,
    use_pdl: bool,
    grid: Optional[int] = None,
    early_rows: int = DEFAULT_EARLY_ROWS,
    trigger_dependents: bool = False,
    trigger_late: bool = False,
) -> None:
    """Launch the streaming o_proj kernel on the current stream.

    Args:
        v: ``[num_rows, K]`` contiguous BF16, ``num_rows <= 16``.
        w_o: ``[N, K]`` contiguous BF16, 16-byte aligned.
        out: ``[out_rows, N]`` contiguous BF16; the kernel writes rows
            ``[0, min(out_rows, 16))`` (rows past ``num_rows`` as zeros).
        use_pdl: launch as a PDL dependent of the previous kernel.
        grid: CTA count (default: the device's SM count).
        early_rows: with ``use_pdl``, rows of ``w_o`` per CTA copied before
            ``griddepcontrol.wait`` (the rest follow the wait).
        trigger_dependents: execute ``griddepcontrol.launch_dependents`` once
            all row copies are issued, so a PDL-launched consumer can start
            its prologue; it must still wait for this grid before reading.
        trigger_late: with ``trigger_dependents``, trigger only after the
            CTA's MMAs instead (experiments).
    """
    n_dim, k_dim = w_o.shape
    if grid is None:
        grid = _sm_count(w_o.device.index or 0)
    n_tiles_r = 1 if out.shape[0] <= 8 else 2
    if not use_pdl:
        early_rows = 1 << 30
    key = (n_dim, k_dim, grid, n_tiles_r, use_pdl, early_rows, trigger_dependents, trigger_late)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = _build(
            n_dim, k_dim, grid, n_tiles_r, use_pdl, early_rows, trigger_dependents, trigger_late
        )
        _COMPILED[key] = compiled
    compiled(v, w_o, out)
