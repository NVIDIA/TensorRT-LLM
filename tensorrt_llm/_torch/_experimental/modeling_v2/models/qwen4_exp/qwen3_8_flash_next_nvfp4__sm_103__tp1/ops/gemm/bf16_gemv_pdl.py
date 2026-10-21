# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-row bf16 projection launched with programmatic dependent launch (PDL),
its weight streaming in while the kernel before it finishes.

Tier 5 (CuTe DSL written for this target). At decode row counts a projection
reads its weight once and its activation is a few KB, so the GEMV's time is the
weight stream plus its launch and ramp. A normal launch cannot read a byte of
that stream until the producer of the activation has exited. Launched with PDL,
every CTA loads its weight rows -- static, so safe to read early -- into
registers before `griddepcontrol.wait` and triggers its own dependents right
before it; the activation is read, and the output written, only after the wait.

One CTA computes `OUTS` output columns over the whole K for every row; element
`k` of a row is held by thread `(k // VEC) % THREADS` as its vector
`(k // VEC) // THREADS`, so each k-tile of loads is contiguous across a warp.
Products and sums in fp32: a warp reduction, then a cross-warp one in shared
memory; one rounding at the store.

A K with a shared-memory tier (`SOUTS`) gives each CTA `SOUTS` more outputs whose
weight rows it stages in shared memory with cp.async before the wait, each thread
exactly the vectors it later reads (so `cp.async.wait_group` alone orders them). The
2560 x 6144 projection then runs as one wave of 640 CTAs whose whole weight is on
chip by the wait, instead of 1280 CTAs of which the last third start loading theirs
only as the first ones exit.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda_drv
import cutlass
import cutlass.cute as cute
import torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

#: Largest row count one launch serves; every row re-reads the weight from registers.
MAX_ROWS = 4
#: (threads, outputs per CTA, elements per vector load) per K. Swept on GB300 at one row,
#: graph replay over per-layer weights: K=2560 64 threads x 2 outputs; K=640 (80 16-byte
#: vectors, which no warp multiple divides) 32 threads x 4 outputs on 8-byte vectors.
TILING = {2560: (64, 2, 8), 6144: (128, 2, 8), 640: (32, 4, 4)}
#: Outputs per CTA whose weight rows sit in shared memory (beside the `TILING` ones in
#: registers). K=6144 swept behind the GDN recurrence and the sparse-attention merge
#: (one row, graph replay): 2 + 2 a CTA beat 2 + 0, 2 + 6, 4 + 4 and 1 + 3.
SOUTS = {6144: 2}
_COMPILE_OPTIONS = "--ptxas-options -maxrregcount=64"
_COMPILE_OPTIONS_SMEM = "--ptxas-options -maxrregcount=80"
#: CTAs a multiprocessor must hold for a grid with a shared-memory tier: four, the
#: 640-CTA grid's share of 160 SMs. Behind the sparse-attention merge (one row, graph
#: replay) 4 runs the attention tail 13.5 us a layer against 14.6 at 1 and 15.6 at 2-3;
#: the GDN tail is unchanged.
MIN_BLOCKS_SMEM = 4


def tiling(n: int, k: int) -> tuple[int, int, int, int]:
    """`(threads, outs, souts, vec)` for a `[*, K] x [K, N]` product: register and shared-memory
    outputs a CTA. Without a shared-memory tier that divides N the outputs shrink to divide N."""
    threads, outs, vec = TILING[k]
    souts = SOUTS.get(k, 0)
    if souts and n % (outs + souts) == 0:
        return threads, outs, souts, vec
    while n % outs:
        outs //= 2
    return threads, outs, 0, vec


#: Before its PDL wait a CTA also has the weight rows of CTA `bidx + dist` brought into
#: L2: behind a latency-bound kernel only part of a large grid's CTAs fit, and the later
#: waves' weights would otherwise start streaming only after it. `dist` is about the CTAs
#: resident before the wait: 1216 behind the hyper-connection kernels (measured), at
#: least 912 (a 1280-CTA out projection's first wave) otherwise. Grids smaller than `dist`
#: prefetch nothing.
L2_DIST = 1216
L2_DIST_MIN = 912


def l2_dist(grid: int) -> int:
    return min(L2_DIST, max(grid // 2, L2_DIST_MIN))


@dsl_user_op
def _bulk_prefetch_l2(src, nbytes, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(src.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip),
            cutlass.Int32(nbytes).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class _GemvKernel:
    def __init__(self, *, rows, k, threads, outs, vec, pdl, dist=L2_DIST, souts=0):
        assert k % (threads * vec) == 0 and threads % 32 == 0
        self.rows, self.threads, self.outs, self.souts, self.vec = rows, threads, outs, souts, vec
        self.dist = dist
        self.k_tiles = k // (threads * vec)
        self.warps = threads // 32
        self.pdl = pdl

    def smem_bytes(self) -> int:
        red = self.rows * (self.outs + self.souts) * self.warps * 4
        if not self.souts:
            return red
        w = self.souts * self.k_tiles * self.threads * self.vec * 2
        return -(-w // 128) * 128 + -(-red // 128) * 128 + 128

    @cute.jit
    def __call__(self, mA, mW, mOut, stream: cuda_drv.CUstream):
        n = cute.size(mW, mode=[0])
        load_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.BFloat16,
            num_bits_per_copy=16 * self.vec,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        load_w = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.BFloat16,
            num_bits_per_copy=16 * self.vec,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=16 * self.vec
        )
        self.kernel(mA, mW, mOut, load_a, load_w, g2s).launch(
            grid=[cute.ceil_div(n, self.outs + self.souts), 1, 1],
            block=[self.threads, 1, 1],
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=self.pdl,
            min_blocks_per_mp=MIN_BLOCKS_SMEM if self.souts else 1,
        )

    @cute.kernel
    def kernel(
        self, mA, mW, mOut, load_a: cute.CopyAtom, load_w: cute.CopyAtom, g2s: cute.CopyAtom
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = cute.arch.warp_idx()
        rows: cutlass.Constexpr = self.rows
        outs: cutlass.Constexpr = self.outs
        threads: cutlass.Constexpr = self.threads
        vec: cutlass.Constexpr = self.vec
        k_tiles: cutlass.Constexpr = self.k_tiles
        warps: cutlass.Constexpr = self.warps
        souts: cutlass.Constexpr = self.souts
        tot: cutlass.Constexpr = outs + souts
        n0 = bidx * tot

        # (row, (vec, (threads, k_tiles))): thread t's vectors of a row, k-tile by k-tile.
        w_vec = cute.logical_divide(cute.logical_divide(mW, (None, vec)), (None, (None, threads)))
        a_vec = cute.logical_divide(cute.logical_divide(mA, (None, vec)), (None, (None, threads)))
        smem = cutlass.utils.SmemAllocator()

        # The weight is static: it streams before the wait, while the producer runs. The
        # shared-memory tier first ([output, k-tile, thread, vec]: a warp's vectors are
        # consecutive), so it lands while the register tier streams.
        if const_expr(souts > 0):
            s_w = smem.allocate_tensor(
                cutlass.BFloat16,
                cute.make_layout(
                    (souts, k_tiles, threads, vec),
                    stride=(k_tiles * threads * vec, threads * vec, vec, 1),
                ),
                byte_alignment=128,
            )
            for j in cutlass.range_constexpr(souts):
                s_row = w_vec[n0 + outs + j, (None, (tidx, None))]
                for kt in cutlass.range_constexpr(k_tiles):
                    cute.copy(g2s, s_row[None, kt], s_w[j, kt, tidx, None])
            cute.arch.cp_async_commit_group()
        w_reg = cute.make_rmem_tensor(
            cute.make_layout((outs, k_tiles, vec), stride=(k_tiles * vec, vec, 1)), cutlass.BFloat16
        )
        for j in cutlass.range_constexpr(outs):
            w_row = w_vec[n0 + j, (None, (tidx, None))]
            for kt in cutlass.range_constexpr(k_tiles):
                cute.copy(load_w, w_row[None, kt], w_reg[j, kt, None])
        if const_expr(self.pdl):
            n_rows = cute.size(mW, mode=[0])
            k_len = cute.size(mW, mode=[1])
            if tidx == 0:
                if (bidx + self.dist + 1) * tot <= n_rows:
                    _bulk_prefetch_l2(
                        mW.iterator + (bidx + self.dist) * tot * k_len, tot * k_len * 2
                    )
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()

        acc = cute.make_rmem_tensor(cute.make_layout((rows, tot), stride=(tot, 1)), cutlass.Float32)
        acc.fill(0.0)
        a_reg = cute.make_rmem_tensor(
            cute.make_layout((k_tiles, vec), stride=(vec, 1)), cutlass.BFloat16
        )
        sw_reg = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16)
        for r in cutlass.range_constexpr(rows):
            a_row = a_vec[r, (None, (tidx, None))]
            for kt in cutlass.range_constexpr(k_tiles):
                cute.copy(load_a, a_row[None, kt], a_reg[kt, None])
            for kt in cutlass.range_constexpr(k_tiles):
                for v in cutlass.range_constexpr(vec):
                    av = a_reg[kt, v].to(cutlass.Float32)
                    for j in cutlass.range_constexpr(outs):
                        acc[r, j] = acc[r, j] + av * w_reg[j, kt, v].to(cutlass.Float32)
            if const_expr(souts > 0):
                if const_expr(r == 0):
                    cute.arch.cp_async_wait_group(0)
                for j in cutlass.range_constexpr(souts):
                    for kt in cutlass.range_constexpr(k_tiles):
                        cute.autovec_copy(s_w[j, kt, tidx, None], sw_reg)
                        for v in cutlass.range_constexpr(vec):
                            acc[r, outs + j] = acc[r, outs + j] + a_reg[kt, v].to(
                                cutlass.Float32
                            ) * sw_reg[v].to(cutlass.Float32)
        for r in cutlass.range_constexpr(rows):
            for j in cutlass.range_constexpr(tot):
                acc[r, j] = cute.arch.warp_reduction_sum(acc[r, j])
        part = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((rows, tot, warps), stride=(tot * warps, warps, 1)),
            byte_alignment=16,
        )
        with cute.arch.elect_one():
            for r in cutlass.range_constexpr(rows):
                for j in cutlass.range_constexpr(tot):
                    part[r, j, warp] = acc[r, j]
        cute.arch.sync_threads()
        if tidx < tot:
            for r in cutlass.range_constexpr(rows):
                total = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(warps):
                    total = total + part[r, tidx, w]
                mOut[r, n0 + tidx] = total.to(cutlass.BFloat16)


@functools.lru_cache(maxsize=None)
def _compiled(rows, n, k, pdl):
    threads, outs, souts, vec = tiling(n, k)
    kern = _GemvKernel(
        rows=rows,
        k=k,
        threads=threads,
        outs=outs,
        vec=vec,
        pdl=pdl,
        dist=l2_dist(-(-n // (outs + souts))),
        souts=souts,
    )
    a = torch.empty(rows, k, dtype=torch.bfloat16, device="cuda")
    w = torch.empty(n, k, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(rows, n, dtype=torch.bfloat16, device="cuda")
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(
        kern,
        from_dlpack(a, assumed_align=32),
        from_dlpack(w, assumed_align=32),
        from_dlpack(out, assumed_align=32),
        stream,
        options=_COMPILE_OPTIONS_SMEM if souts else _COMPILE_OPTIONS,
    )


def launch(mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool) -> torch.Tensor:
    """The launch behind `Bf16GemvPdl.__call__`. `pdl` is honoured unless
    `TRTLLM_ENABLE_PDL=0`."""
    m, k = mat_a.shape
    n = mat_b.shape[1]
    dev = mat_a.device
    pdl = pdl and get_env_enable_pdl()
    out = torch.empty((m, n), dtype=mat_a.dtype, device=dev)
    if m > 0:
        fn = _compiled(m, n, k, bool(pdl))
        stream = cuda_drv.CUstream(torch.cuda.current_stream(dev).cuda_stream)
        fn(
            from_dlpack(mat_a, assumed_align=32),
            from_dlpack(mat_b.t(), assumed_align=32),
            from_dlpack(out, assumed_align=32),
            stream,
        )
    return out


def _cell(why: str, m: int, n: int, k: int, pdl: bool = True) -> Cell:
    return Cell(why=why, spec=dict(m=m, n=n, k=k, pdl=pdl))


class Bf16GemvPdl(OpWrapper):
    """`mat_a @ mat_b` in bf16 with fp32 accumulation, one PDL launch per call.

    A drop-in for `ops/gemm/low_m_bf16_mm` at decode row counts: `mat_a` is
    dense row-major `[M, K]` with `M <= MAX_ROWS`, `mat_b` the dense column-major
    `[K, N]` view of a stored `[N, K]` weight (its `.t()`), the result a fresh
    row-major `[M, N]` bf16 tensor. With `pdl` (and PDL available) the kernel is
    launched as a programmatic dependent: it may start before the kernel ahead of
    it on the stream exits, reads only its weight until `griddepcontrol.wait`,
    and lets the kernel after it launch early as well.
    """

    ARCHS = frozenset({Arch.SM_103})

    # Every shape is a qwen3.8-flash-next-nvfp4 tp1 decode projection; the target calls
    # it at one row (`_decode_projection`). Four cells: each `TILING` key is its own
    # compiled kernel the decode forward reaches, plus the N=1 edge.
    CELLS: tuple[Cell, ...] = (
        _cell("qwen3.8 decode M=1 GDN out / attention o, K=6144 shared-memory tier", 1, 2560, 6144),
        _cell("qwen3.8 decode M=1 GDN in-proj concat N=16480, L2 prefetch grid", 1, 16480, 2560),
        _cell("qwen3.8 decode M=1 shared down K=640", 1, 2560, 640),
        _cell("qwen3.8 decode M=1 shared gate N=1, outputs per CTA halved to 1", 1, 1, 2560),
    )

    note = """
    Metadata consumed: none. Preconditions: bf16 operands; `mat_a` dense
    row-major `[M, K]`, `1 <= M <= MAX_ROWS`; `mat_b` the dense column-major
    `[K, N]` view of a `[N, K]` weight (stride `(1, K)`); K a `TILING` key.

    PDL: with `pdl` the launch carries the programmatic-dependent attribute.
    Every read of `mat_a` and the store of the output come after
    `griddepcontrol.wait`, so a producer launched before this kernel is
    complete when they happen, whatever that producer is. Under a CUDA graph a
    cross-stream dependency is an ordinary edge and stays a full dependency.

    Compilation: CuTe DSL, JIT on the first call per (M, N, K, pdl); that call
    must happen before CUDA-graph capture -- the engine's warmup forward at each
    captured batch size does it.

    Numerics: fp32 products and sums, one rounding to bf16; only the summation
    order differs from the fp32 reference.

    Hard limits
      M     rows per launch          boundary MAX_ROWS=4      driven at 1 (cells)
      K     a TILING key             2560, 6144, 640          all driven
      N     any; the outputs per CTA halve until they divide N    N=1 driven

    The test also replays a PDL chain of these GEMVs under a CUDA graph, each
    reading the one before it.

    Not certified: M = 2..4; PDL off (`pdl=False`, or PDL disabled in the
    environment), the same kernel without the attribute.
    """

    def __call__(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = True) -> torch.Tensor:
        return launch(mat_a, mat_b, pdl)

    def reference(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = True) -> torch.Tensor:
        """fp32-accumulated, TF32 off; `pdl` is a launch attribute and is ignored."""
        with true_fp32_matmul():
            return (mat_a.float() @ mat_b.float()).to(mat_a.dtype)

    def is_valid(self, mat_a: torch.Tensor, mat_b: torch.Tensor, pdl: bool = True) -> None:
        assert mat_a.dtype == torch.bfloat16 and mat_b.dtype == torch.bfloat16, "bf16 operands only"
        assert mat_a.dim() == 2 and mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
        assert 1 <= mat_a.shape[0] <= MAX_ROWS, f"M={mat_a.shape[0]} outside 1..{MAX_ROWS}"
        assert mat_b.shape[0] == mat_a.shape[1], "inner dimensions differ"
        assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
            "mat_b must be dense column-major [K, N] (e.g. weight.t())"
        )
        assert mat_a.shape[1] in TILING, f"K={mat_a.shape[1]} has no tiling"


bf16_gemv_pdl = Bf16GemvPdl()
