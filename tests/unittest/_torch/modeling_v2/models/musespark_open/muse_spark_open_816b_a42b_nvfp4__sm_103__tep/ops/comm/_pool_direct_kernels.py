# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stand-in kernels of the ``pool_into_route`` chain test: the attention hop and the expert kernel.

``hop_standin(src, dst, pre_ns, post_ns, stamp)`` -- the attention hop as the
pool GEMM and the forked router see it (the consumer hop with its early trigger):
``CTAS`` CTAs of 128 threads launched with programmatic dependent launch; each
waits for its predecessors, spins ``pre_ns`` (the hop's poll), triggers its
dependents (the early-trigger placement after the last transport access), spins
``post_ns`` (the boundary norms), then writes the layer's input rows
``src -> dst``; thread 0 of CTA 0 then stores the global timer at ``stamp``.

``moe_standin(fp4, stamp, spin_ns, sink)`` -- the expert kernel's start: PDL,
triggers its dependents at entry, waits, thread 0 of CTA 0 stores the global
timer at ``stamp`` (the moment the routed kernel's outputs are visible), every
CTA reads one word of ``fp4`` and spins ``spin_ns``, then one store to ``sink``.

Every CTA of both waits before it exits.
"""

import functools

import torch

_T = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)

HOP_CTAS = 64
HOP_THREADS = 128
MOE_CTAS = 64
MOE_THREADS = 256


@functools.lru_cache(maxsize=None)
def _hop(device_index: int, pre_ns: int, post_ns: int):
    import importlib

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cutlass_dsl import Int32, Int64

    ptx = importlib.import_module(_T + ".ops.moe._persistent_experts.ptx")

    class _Hop:
        @cute.jit
        def run(self, src, dst, words, stamp, stream):
            self.kernel(src, dst, words, stamp).launch(
                grid=[HOP_CTAS, 1, 1], block=[HOP_THREADS, 1, 1], stream=stream, use_pdl=True
            )

        @cute.kernel
        def kernel(self, src: Int64, dst: Int64, words: Int32, stamp: Int64):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(pre_ns > 0):
                t0 = ptx.globaltimer_u64()
                t = t0
                while t - t0 < Int64(pre_ns):
                    t = ptx.globaltimer_u64()
            cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(post_ns > 0):
                t0 = ptx.globaltimer_u64()
                t = t0
                while t - t0 < Int64(post_ns):
                    t = ptx.globaltimer_u64()
            i = Int32(bidx * HOP_THREADS + tidx)
            while i < words:
                ptx.st_u32(dst + Int64(i) * 4, ptx.ld_u32(src + Int64(i) * 4))
                i = i + HOP_CTAS * HOP_THREADS
            cute.arch.barrier()
            if bidx == 0:
                if tidx == 0:
                    ptx.st_u64(stamp, ptx.globaltimer_u64())

    with torch.cuda.device(device_index):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute.compile(
            _Hop().run,
            cutlass.Int64(0),
            cutlass.Int64(0),
            cutlass.Int32(0),
            cutlass.Int64(0),
            stream,
        )


def hop_standin(
    src: torch.Tensor, dst: torch.Tensor, pre_ns: int, post_ns: int, stamp: int
) -> None:
    import cutlass
    from cuda.bindings import driver as cuda

    assert src.is_contiguous() and dst.is_contiguous() and src.nbytes == dst.nbytes
    fn = _hop(torch.cuda.current_device(), int(pre_ns), int(post_ns))
    fn(
        cutlass.Int64(src.data_ptr()),
        cutlass.Int64(dst.data_ptr()),
        cutlass.Int32(src.nbytes // 4),
        cutlass.Int64(int(stamp)),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


@functools.lru_cache(maxsize=None)
def _moe(device_index: int, spin_ns: int):
    import importlib

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cutlass_dsl import Int64

    ptx = importlib.import_module(_T + ".ops.moe._persistent_experts.ptx")

    class _Moe:
        @cute.jit
        def run(self, fp4, stamp, sink, stream):
            self.kernel(fp4, stamp, sink).launch(
                grid=[MOE_CTAS, 1, 1], block=[MOE_THREADS, 1, 1], stream=stream, use_pdl=True
            )

        @cute.kernel
        def kernel(self, fp4: Int64, stamp: Int64, sink: Int64):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
            if bidx == 0:
                if tidx == 0:
                    ptx.st_u64(stamp, ptx.globaltimer_u64())
            value = ptx.ld_u32(fp4 + Int64(tidx % 32) * 4)
            if cutlass.const_expr(spin_ns > 0):
                t0 = ptx.globaltimer_u64()
                t = t0
                while t - t0 < Int64(spin_ns):
                    t = ptx.globaltimer_u64()
            if tidx == 0:
                ptx.st_u32(sink + Int64(bidx) * 4, value)

    with torch.cuda.device(device_index):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        return cute.compile(
            _Moe().run, cutlass.Int64(0), cutlass.Int64(0), cutlass.Int64(0), stream
        )


def moe_standin(fp4: torch.Tensor, stamp: int, spin_ns: int, sink: torch.Tensor) -> None:
    import cutlass
    from cuda.bindings import driver as cuda

    assert sink.numel() >= MOE_CTAS and sink.dtype == torch.int32
    fn = _moe(torch.cuda.current_device(), int(spin_ns))
    fn(
        cutlass.Int64(fp4.data_ptr()),
        cutlass.Int64(int(stamp)),
        cutlass.Int64(sink.data_ptr()),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
