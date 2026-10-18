# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test kernels of ``early_expert_ids``: a late PDL producer, optionally publishing tagged ids.

``late_producer(...)`` launches one 256-thread CTA with programmatic dependent launch
on the current stream. In order it

1. triggers its dependents (``griddepcontrol.launch_dependents``) at entry, so the
   next kernel may become resident at once;
2. waits for its own predecessors (``griddepcontrol.wait``);
3. with ``publish``: stores the 64 tagged id words of an ``EarlyExpertIds`` buffer
   from ``ids`` (rows past ``n_ids`` get -1), tagged with the expert workspace's
   counter + 1 (0 skipped), exactly as ``pool_gather_route`` does;
4. spins ``delay_ns`` on the global timer;
5. copies up to three word buffers ``src -> dst`` (the consumer's real inputs).

So a consumer that reads its inputs before its own dependency wait reads the
destination buffers before step 5 wrote them, while the published ids are fresh.
Every CTA waits before it exits, as every kernel between two expert launches must.
"""

import functools

import torch

_T = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open."
    "muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)

THREADS = 256
WORDS = 64


@functools.lru_cache(maxsize=None)
def _compiled(device_index: int, delay_ns: int, publish: int, epoch: int):
    import importlib

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cutlass_dsl import Int32, Int64, Uint32

    ptx = importlib.import_module(_T + ".ops.moe._persistent_experts.ptx")

    class _LateProducer:
        @cute.jit
        def run(self, s0, d0, n0, s1, d1, n1, s2, d2, n2, ids, n_ids, stream):
            self.kernel(s0, d0, n0, s1, d1, n1, s2, d2, n2, ids, n_ids).launch(
                grid=[1, 1, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=True
            )

        @cute.jit
        def _copy(self, tidx, src, dst, n):
            i = Int32(tidx)
            while i < n:
                ptx.st_u32(dst + Int64(i) * 4, ptx.ld_u32(src + Int64(i) * 4))
                i = i + THREADS

        @cute.kernel
        def kernel(
            self,
            s0: Int64,
            d0: Int64,
            n0: Int32,
            s1: Int64,
            d1: Int64,
            n1: Int32,
            s2: Int64,
            d2: Int64,
            n2: Int32,
            ids: Int64,
            n_ids: Int32,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(publish != 0):
                if tidx < WORDS:
                    tag = ptx.ld_relaxed_gpu_u32(Int64(epoch)) + Uint32(1)
                    if tag == Uint32(0):
                        tag = Uint32(1)
                    v = Uint32(0xFFFFFFFF)
                    if tidx < n_ids:
                        v = ptx.ld_u32(ids + Int64(tidx) * 4)
                    ptx.st_relaxed_gpu_u64_pair(Int64(publish) + Int64(tidx) * 8, v, tag)
            if cutlass.const_expr(delay_ns > 0):
                t0 = ptx.globaltimer_u64()
                t = t0
                while t - t0 < Int64(delay_ns):
                    t = ptx.globaltimer_u64()
            self._copy(tidx, s0, d0, n0)
            self._copy(tidx, s1, d1, n1)
            self._copy(tidx, s2, d2, n2)

    with torch.cuda.device(device_index):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        zero64 = [cutlass.Int64(0)] * 2
        return cute.compile(
            _LateProducer().run,
            *zero64,
            cutlass.Int32(0),
            *zero64,
            cutlass.Int32(0),
            *zero64,
            cutlass.Int32(0),
            cutlass.Int64(0),
            cutlass.Int32(0),
            stream,
        )


def late_producer(copies, delay_ns: int = 0, early=None, ids=None) -> None:
    """Launch the late producer on the current stream (see the module docstring).

    ``copies``: up to three ``(src, dst)`` tensor pairs (same byte size, multiples of 4
    bytes); ``early``: an ``EarlyExpertIds`` to publish ``ids`` (an int32 tensor) into.
    """
    import cutlass
    from cuda.bindings import driver as cuda

    device = torch.cuda.current_device()
    publish = early.words_address if early is not None else 0
    epoch = early.epoch_address if early is not None else 0
    fn = _compiled(device, int(delay_ns), publish, epoch)
    args = []
    pairs = list(copies) + [None] * (3 - len(copies))
    for pair in pairs:
        if pair is None:
            args += [cutlass.Int64(0), cutlass.Int64(0), cutlass.Int32(0)]
            continue
        src, dst = pair
        assert src.numel() * src.element_size() == dst.numel() * dst.element_size()
        assert src.is_contiguous() and dst.is_contiguous()
        words = src.numel() * src.element_size() // 4
        args += [cutlass.Int64(src.data_ptr()), cutlass.Int64(dst.data_ptr()), cutlass.Int32(words)]
    n_ids = 0 if ids is None else ids.numel()
    args += [cutlass.Int64(0 if ids is None else ids.data_ptr()), cutlass.Int32(n_ids)]
    fn(*args, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
