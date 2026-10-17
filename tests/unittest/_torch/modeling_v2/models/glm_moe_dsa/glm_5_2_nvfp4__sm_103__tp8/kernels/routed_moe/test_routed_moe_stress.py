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
"""Stress, scheduling and real-input tests of the routed_moe cross-CTA protocol (sm_103).

* ``test_sanitizer_cells``: routed_moe-vs-native cells in both output modes plus a
  short graph, meant to be run under ``compute-sanitizer`` (memcheck /
  synccheck / racecheck, kernel filter ``kns=RoutedMoeKernel``) in both
  FC1->FC2 hand-off modes.
* ``test_delayed_*``: the kernel's delay injection (``routed_moe.DELAY_INJECTION``) holds a
  pseudo-random third of the CTAs (re-drawn every launch) at CTA entry, before
  publishing FC1 tiles and split partials, before split owners gather and
  before FC2 operand fetches; outputs must stay bit-identical to undelayed
  launches.
* ``test_corun_*``: other kernels on a second stream hold SMs while routed_moe runs
  (a shared-memory-heavy spinning kernel, cuBLAS GEMMs), so not all routed_moe CTAs
  are co-resident; routed_moe must complete with bit-identical outputs.
* ``test_graph_many_launches_real_layers``: one CUDA graph with dozens of
  back-to-back routed_moe launches over several real layers (one shared workspace),
  every output equal to eager routed_moe and classified against native.
* ``test_real_*``: a native capture of real routed-MoE inputs.

Comparisons with native use ``routed_moe_fixtures.classify_vs_native`` (exact
expert IDs and scales; finalized output exact, FP32-reorder-close or within
the frozen budget).

Layers: ``GLM_MOE_DSA_ROUTED_MOE_REAL_LAYERS`` (default ``10,30,50,77``, the layers with captured
hidden states) of TP8 rank ``GLM_MOE_DSA_ROUTED_MOE_RANK``.
"""

import faulthandler
import os
import sys
import time

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import pytest
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import Int32, Int64, T, Uint32, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
    ptx,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import routed_moe_fixtures as fx  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)

RANK = int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_RANK", "0"))
LAYERS = [
    int(v) for v in os.environ.get("GLM_MOE_DSA_ROUTED_MOE_REAL_LAYERS", "10,30,50,77").split(",")
]
WATCHDOG_S = 180


@pytest.fixture(scope="module")
def layers():
    return {lay: fx.build_native_layer(layer=lay, tp_rank=RANK) for lay in LAYERS}


@pytest.fixture(scope="module")
def runners(layers):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        RoutedMoe,
    )

    return {lay: RoutedMoe.from_native(m.module, m.bias) for lay, m in layers.items()}


@pytest.fixture(scope="module")
def decoders(layers):
    probe = layers[LAYERS[0]].hidden_proxy(fx.prompt_token_ids()[:1])[0]
    return {lay: fx.ExpertDecoder(m, probe) for lay, m in layers.items()}


@pytest.fixture(scope="module")
def captures():
    return {
        lay: [(None if h is None else h.cuda(), lg.cuda()) for h, lg in fx.load_real_capture(lay)]
        for lay in LAYERS
    }


def _synthetic(layer, rows, distinct, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(rows, fx.HIDDEN, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    sets = fx.expert_sets_with_distinct(rows, distinct, seed=seed)
    return x, fx.logits_for_expert_sets(sets, layer.bias, seed=seed)


def _real(captures, lay, rows, index):
    win = fx.real_windows(captures[lay], rows)
    x, lg = win[(index * 131) % len(win)]
    return x.cuda().contiguous(), lg.cuda().contiguous()


CELLS = [(1, 8), (2, 14), (4, 25), (6, 21), (6, 38), (8, 48)]


def _cells(layer, captures, lay, seed=0):
    out = [(f"R{r}D{d}", *_synthetic(layer, r, d, seed + r * 100 + d)) for r, d in CELLS]
    for i in range(3):
        out.append((f"real{lay}R6#{i}", *_real(captures, lay, 6, seed + i)))
    out.append((f"real{lay}R8", *_real(captures, lay, 8, seed + 7)))
    return out


def _identical(a, b, label):
    """routed_moe against routed_moe: bit-identical finalized outputs or raw tuples."""
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), label
    else:
        for i in range(3):
            assert torch.equal(a[i], b[i]), f"{label} (raw part {i})"


def _routed_moe_both(runner, x, lg):
    return runner(x, lg), runner(x, lg, do_finalize=False)


def _classify(layer, x, lg, fin, raw, label, emul=True):
    n_fin = layer.native_forward(x, lg)
    n_raw = layer.native_forward(x, lg, do_finalize=False)
    emul_fn = (lambda: fx.emulate_moe(layer, x, lg)) if emul else None
    return fx.classify_vs_native(fin, raw, n_fin, n_raw, emul_fn, label)


# ---------------------------------------------------------------------------
# (a) compute-sanitizer target
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["fin", "raw"])
def test_sanitizer_cells(layers, runners, captures, mode):
    """One output mode per run, so a sanitizer report names one compiled
    kernel. Outputs are compared with native loosely here (the exact checks
    are in the other tests). The graph segment needs the caching allocator
    (``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` would cudaMalloc during capture).
    """
    lay = LAYERS[0]
    layer, runner = layers[lay], runners[lay]
    fin = mode == "fin"
    for label, x, lg in _cells(layer, captures, lay):
        out = runner(x, lg, do_finalize=fin)
        ref = layer.native_forward(x, lg, do_finalize=fin)
        torch.cuda.synchronize()
        if fin:
            assert fx.error_stats(out, ref)["nrms"] < 2e-3, label
        else:
            rep = fx.raw_report(out, ref)
            assert rep["scale_exact"] and rep["same_expert"], (label, rep)
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1":
        return
    x, lg = _real(captures, lay, 6, 11)
    eager = runner(x, lg, do_finalize=fin)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = [runner(x, lg, do_finalize=fin) for _ in range(4)]
    graph.replay()
    torch.cuda.synchronize()
    for i, o in enumerate(outs):
        _identical(o, eager, f"graph launch {i}")


class _BulkCopyControl:
    """Racecheck control: the bulk-copy -> mbarrier (transaction count) ->
    shared-load pattern routed_moe uses for its activation rows and FC2 operand
    images, in isolation and correctly synchronized. Two phases on one
    buffer: copy, wait, every thread reads, CTA barrier, copy again (the
    write-after-read reuse), wait, read.
    """

    NBYTES = 2048

    @cute.jit
    def run(self, src: Int64, dst: Int64, stream):
        @cute.struct
        class Storage:
            mbar: cute.struct.MemRange[cutlass.Int64, 2]
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2048], 128]

        self.storage_t = Storage
        self.kernel(src, dst).launch(
            grid=[1, 1, 1], block=[128, 1, 1], smem=Storage.size_in_bytes(), stream=stream
        )

    @cute.kernel
    def kernel(self, src: Int64, dst: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage_t)
        bar = st.mbar.data_ptr()
        buf = Int32(st.buf.data_ptr().toint())
        if tidx == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init(bar + 1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        for phase in cutlass.range_constexpr(2):
            if tidx == 0:
                b = Int32((bar + phase).toint())
                ptx.mbar_arrive_expect_tx(b, Int32(self.NBYTES))
                ptx.bulk_g2s(buf, src + Int64(phase * self.NBYTES), Int32(self.NBYTES), b)
            cute.arch.mbarrier_wait(bar + phase, 0)
            for j in cutlass.range_constexpr(4):
                v = ptx.lds_u32(buf + (j * 128 + tidx) * 4)
                ptx.st_u32(dst + Int64(phase * self.NBYTES) + Int64((j * 128 + tidx) * 4), v)
            # generic reads before the next async-proxy write of the buffer
            ptx.fence_proxy_async_shared()
            cute.arch.sync_threads()


class _ChainControl:
    """Racecheck control for routed_moe's shared-buffer reuse (FC1 B operand stages
    overwritten by FC2 operands): warp 0 writes a buffer with generic stores,
    proxy fence, mbarrier arrive; warp 1 waits and signals warp 2 through a
    second mbarrier, with a plain arrive or with ``tcgen05.commit`` (routed_moe's
    FC1-done signal); warp 2 waits, overwrites the buffer with a bulk copy,
    waits for it and reads it back. Correctly ordered in both variants.
    """

    NBYTES = 1024

    def __init__(self, use_commit: bool):
        self.use_commit = use_commit

    @cute.jit
    def run(self, src: Int64, dst: Int64, stream):
        @cute.struct
        class Storage:
            mbar: cute.struct.MemRange[cutlass.Int64, 3]
            buf: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 1024], 128]

        self.storage_t = Storage
        self.kernel(src, dst).launch(
            grid=[1, 1, 1], block=[96, 1, 1], smem=Storage.size_in_bytes(), stream=stream
        )

    @cute.kernel
    def kernel(self, src: Int64, dst: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        smem = utils.SmemAllocator()
        st = smem.allocate(self.storage_t)
        bar = st.mbar.data_ptr()
        buf = Int32(st.buf.data_ptr().toint())
        if tidx == 0:
            cute.arch.mbarrier_init(bar, 1)
            cute.arch.mbarrier_init(bar + 1, 1)
            cute.arch.mbarrier_init(bar + 2, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp == 0:
            for j in cutlass.range_constexpr(8):
                ptx.sts_u32(buf + (j * 32 + lane) * 4, Uint32(lane + 32 * j))
            ptx.fence_proxy_async_shared()
            cute.arch.sync_warp()
            if lane == 0:
                cute.arch.mbarrier_arrive(bar)
        if warp == 1:
            cute.arch.mbarrier_wait(bar, 0)
            if cutlass.const_expr(self.use_commit):
                with cute.arch.elect_one():
                    tcgen05.commit(bar + 1)
            else:
                if lane == 0:
                    cute.arch.mbarrier_arrive(bar + 1)
        if warp == 2:
            cute.arch.mbarrier_wait(bar + 1, 0)
            if lane == 0:
                b2 = Int32((bar + 2).toint())
                ptx.mbar_arrive_expect_tx(b2, Int32(self.NBYTES))
                ptx.bulk_g2s(buf, src, Int32(self.NBYTES), b2)
            cute.arch.mbarrier_wait(bar + 2, 0)
            for j in cutlass.range_constexpr(8):
                v = ptx.lds_u32(buf + (j * 32 + lane) * 4)
                ptx.st_u32(dst + Int64((j * 32 + lane) * 4), v)


@pytest.mark.parametrize("use_commit", [False, True])
def test_racecheck_control_chain(use_commit):
    from cuda.bindings import driver as cuda

    src = torch.randint(0, 255, (_ChainControl.NBYTES,), dtype=torch.uint8, device="cuda")
    dst = torch.zeros_like(src)
    cur = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(_ChainControl(use_commit).run, cutlass.Int64(0), cutlass.Int64(0), cur)
    fn(cutlass.Int64(src.data_ptr()), cutlass.Int64(dst.data_ptr()), cur)
    torch.cuda.synchronize()
    assert torch.equal(src, dst)


def test_racecheck_control_bulk_copy():
    from cuda.bindings import driver as cuda

    src = torch.randint(0, 255, (2 * _BulkCopyControl.NBYTES,), dtype=torch.uint8, device="cuda")
    dst = torch.zeros_like(src)
    cur = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    ctl = _BulkCopyControl()
    fn = cute.compile(ctl.run, cutlass.Int64(0), cutlass.Int64(0), cur)
    fn(cutlass.Int64(src.data_ptr()), cutlass.Int64(dst.data_ptr()), cur)
    torch.cuda.synchronize()
    assert torch.equal(src, dst)


# ---------------------------------------------------------------------------
# (b) delayed producers / owners / consumers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("delay_ns", [2000, 9000])
@pytest.mark.parametrize("finalize", [True, False])
def test_delayed_ctas_exact(layers, runners, captures, monkeypatch, delay_ns, finalize):
    """Delayed launches are bit-identical to undelayed ones (eager and in a
    replayed graph); the undelayed outputs are classified against native.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        routed_moe as routed_moe_mod,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.kernel import (  # noqa: E501
        DELAY_CONSUMER,
        DELAY_ENTRY,
        DELAY_OWNER,
        DELAY_PRODUCER,
    )

    lay = LAYERS[0]
    layer, runner = layers[lay], runners[lay]
    cells = _cells(layer, captures, lay, seed=delay_ns)
    assert routed_moe_mod.DELAY_INJECTION[0] == 0
    undelayed = []
    classes = {}
    for label, x, lg in cells:
        fin, raw = _routed_moe_both(runner, x, lg)
        cls, _ = _classify(layer, x, lg, fin, raw, label)
        classes[cls] = classes.get(cls, 0) + 1
        undelayed.append(fin if finalize else raw)
    torch.cuda.synchronize()
    sites = DELAY_ENTRY | DELAY_PRODUCER | DELAY_OWNER | DELAY_CONSUMER
    monkeypatch.setattr(routed_moe_mod, "DELAY_INJECTION", (sites, delay_ns))
    t0 = time.perf_counter()
    for rep in range(3):  # every launch draws a new delayed set
        for (label, x, lg), base in zip(cells, undelayed):
            out = runner(x, lg, do_finalize=finalize)
            torch.cuda.synchronize()
            _identical(out, base, f"{label} rep{rep} delay={delay_ns}")
    eager_s = time.perf_counter() - t0
    # graph: all cells back to back, replayed (epochs advance per replay)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    static_inputs = [(x.clone(), lg.clone()) for _, x, lg in cells]
    with torch.cuda.stream(stream):
        runner(static_inputs[0][0], static_inputs[0][1], do_finalize=finalize)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = [runner(x, lg, do_finalize=finalize) for x, lg in static_inputs]
    for rep in range(4):
        graph.replay()
        torch.cuda.synchronize()
        for (label, _, _), o, base in zip(cells, outs, undelayed):
            _identical(o, base, f"graph {label} rep{rep}")
    print(
        f"[routed_moe] delayed CTAs bit-identical: delay_ns={delay_ns} finalize={finalize} "
        f"eager {eager_s:.2f}s; undelayed vs native {classes}"
    )


# ---------------------------------------------------------------------------
# (c) co-running kernels that hold SMs
# ---------------------------------------------------------------------------
@dsl_user_op
def _sreg_u32(name: str, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [],
            f"mov.u32 $0, %{name};",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _SmHog:
    """Test-only kernel: ``ctas`` CTAs that each reserve ``smem_bytes`` of
    shared memory (more than half an SM, so one CTA per SM and no routed_moe CTA
    beside it) and spin for ``spin_ns`` on the global timer. Each CTA records
    ``started``, its SM id and its dynamic shared-memory size.
    """

    def __init__(self, ctas: int, smem_bytes: int, spin_ns: int):
        self.ctas = ctas
        self.smem_bytes = smem_bytes
        self.spin_ns = spin_ns

    @cute.jit
    def run(self, info_addr: Int64, stream):
        self.kernel(info_addr).launch(
            grid=[self.ctas, 1, 1], block=[32, 1, 1], smem=self.smem_bytes, stream=stream
        )

    @cute.kernel
    def kernel(self, info_addr: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        rec = info_addr + Int64(bidx) * Int64(16)
        if tidx == 0:
            ptx.st_u32(rec + Int64(4), _sreg_u32("smid"))
            ptx.st_u32(rec + Int64(8), _sreg_u32("dynamic_smem_size"))
            ptx.st_relaxed_gpu_u32(rec, Uint32(1))
        t0 = ptx.globaltimer_u64()
        t = t0
        while t - t0 < Int64(self.spin_ns):
            t = ptx.globaltimer_u64()
        if tidx == 0:
            ptx.st_u32(rec + Int64(12), Uint32(1))


class _WaitStarted:
    """Test-only kernel: one thread spins until ``n`` hog CTAs have started,
    so work enqueued after it on the same stream runs while they are all
    resident.
    """

    def __init__(self, n: int):
        self.n = n

    @cute.jit
    def run(self, info_addr: Int64, stream):
        self.kernel(info_addr).launch(grid=[1, 1, 1], block=[32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, info_addr: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            i = Int32(0)
            while i < self.n:
                if ptx.ld_relaxed_gpu_u32(info_addr + Int64(i) * Int64(16)) != Uint32(0):
                    i = i + 1


def _make_hog(ctas: int, smem_bytes: int, spin_ns: int):
    """Returns ``(launch_hog(stream), wait_started(stream), info)``; ``info``
    is int32 ``[ctas, 4]``: started, SM id, dynamic smem bytes, done.
    """
    from cuda.bindings import driver as cuda

    cur = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    hog_fn = cute.compile(_SmHog(ctas, smem_bytes, spin_ns).run, cutlass.Int64(0), cur)
    wait_fn = cute.compile(_WaitStarted(ctas).run, cutlass.Int64(0), cur)
    info = torch.zeros(ctas, 4, dtype=torch.int32, device="cuda")

    def launch(stream: torch.cuda.Stream):
        hog_fn(cutlass.Int64(info.data_ptr()), cuda.CUstream(stream.cuda_stream))

    def wait_started(stream: torch.cuda.Stream):
        wait_fn(cutlass.Int64(info.data_ptr()), cuda.CUstream(stream.cuda_stream))

    return launch, wait_started, info


class _Watchdog:
    def __enter__(self):
        faulthandler.dump_traceback_later(WATCHDOG_S, exit=True)
        return self

    def __exit__(self, *exc):
        faulthandler.cancel_dump_traceback_later()
        return False


def _refs(layer, runner, cells):
    """Eager routed_moe outputs without co-runners (the reference the co-run
    outputs must equal bit for bit), classified against native.
    """
    refs, classes = [], {}
    for label, x, lg in cells:
        fin, raw = _routed_moe_both(runner, x, lg)
        cls, _ = _classify(layer, x, lg, fin, raw, label)
        classes[cls] = classes.get(cls, 0) + 1
        refs.append(fin)
    torch.cuda.synchronize()
    return refs, classes


def _ev():
    return torch.cuda.Event(enable_timing=True)


def test_corun_sm_hog(layers, runners, captures):
    """routed_moe while a spinning kernel holds SMs (one CTA per SM, more shared
    memory than a routed_moe CTA leaves free). A and C enqueue routed_moe only after every
    hog CTA is resident, so routed_moe starts with part (C) or none (A) of its grid
    resident and must finish once the hog exits; B starts the hog at
    staggered points of routed_moe's execution; D replays a routed_moe graph against hogs.
    Outputs must equal the undisturbed routed_moe outputs bit for bit.
    """
    lay = LAYERS[0]
    layer, runner = layers[lay], runners[lay]
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    smem = 120 * 1024
    hog_all, wait_all, info_all = _make_hog(sms, smem, 300_000)  # every SM, 300 us
    hog_half, wait_half, info_half = _make_hog(sms // 2, smem, 2_000_000)  # half, 2 ms
    cells = _cells(layer, captures, lay, seed=5)
    refs, classes = _refs(layer, runner, cells)
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    timing = {}

    def check(outs, label):
        for i, (o, r) in enumerate(zip(outs, refs)):
            _identical(o, r, f"{label} #{i}")

    def held(hog, wait, info, label):
        """Hog on ``side``; routed_moe batch on ``main`` once every hog CTA runs."""
        info.zero_()
        e0, e_hog, e_first, e_all = _ev(), _ev(), _ev(), _ev()
        side.wait_stream(main)
        e0.record(side)
        hog(side)
        e_hog.record(side)
        main.wait_event(e0)
        wait(main)
        outs = []
        for i, (_, x, lg) in enumerate(cells):
            outs.append(runner(x, lg))
            if i == 0:
                e_first.record(main)
        e_all.record(main)
        torch.cuda.synchronize()
        check(outs, label)
        assert int(info[:, 0].sum()) == info.shape[0] and int(info[:, 3].sum()) == info.shape[0]
        timing[f"{label}_hog_end_ms"] = round(e0.elapsed_time(e_hog), 3)
        timing[f"{label}_first_routed_moe_end_ms"] = round(e0.elapsed_time(e_first), 3)
        timing[f"{label}_all_routed_moe_end_ms"] = round(e0.elapsed_time(e_all), 3)
        timing[f"{label}_hog_sms"] = len(set(info[:, 1].tolist()))
        timing[f"{label}_hog_dyn_smem"] = sorted(set(info[:, 2].tolist()))

    with _Watchdog():
        e0, e1 = _ev(), _ev()
        e0.record(main)
        outs = [runner(x, lg) for _, x, lg in cells]
        e1.record(main)
        torch.cuda.synchronize()
        check(outs, "alone")
        timing["alone_batch_ms"] = round(e0.elapsed_time(e1), 3)
        held(hog_all, wait_all, info_all, "A")
        held(hog_half, wait_half, info_half, "C")
        # B: routed_moe first, the hog launched at staggered offsets
        for delay_cycles in (0, 2000, 10000, 30000, 60000, 120000):
            for rep in range(4):
                info_all.zero_()
                side.wait_stream(main)
                outs = [runner(x, lg) for _, x, lg in cells[:4]]
                with torch.cuda.stream(side):
                    torch.cuda._sleep(delay_cycles)
                hog_all(side)
                outs += [runner(x, lg) for _, x, lg in cells[4:]]
                torch.cuda.synchronize()
                check(outs, f"hog after {delay_cycles} cycles rep{rep}")
                assert int(info_all[:, 3].sum()) == sms
        # D: a captured graph of routed_moe launches replayed against repeated hogs
        static_inputs = [(x.clone(), lg.clone()) for _, x, lg in cells]
        gstream = torch.cuda.Stream()
        gstream.wait_stream(main)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=gstream):
            gouts = [runner(x, lg) for x, lg in static_inputs]
        for rep in range(8):
            side.wait_stream(main)
            if rep % 2 == 0:
                hog_all(side)
                wait_all(main)
            graph.replay()
            if rep % 2 == 1:
                hog_half(side)
            torch.cuda.synchronize()
            check(gouts, f"graph vs hog rep{rep}")
    print(f"[routed_moe] co-run SM hog ok: timing {timing}; refs vs native {classes}")
    # the hogs really held one SM per CTA with their full shared memory
    assert timing["C_hog_sms"] == sms // 2 and timing["A_hog_sms"] == sms, timing
    assert timing["A_hog_dyn_smem"] == [smem] and timing["C_hog_dyn_smem"] == [smem], timing


def test_corun_gemm_side_stream(layers, runners, captures):
    """CuBLAS GEMMs (shared-memory-heavy, many CTAs) on a second stream while
    a graph of routed_moe launches replays, as the shared expert's GEMMs overlap the
    routed experts in the served model.
    """
    lay = LAYERS[0]
    layer, runner = layers[lay], runners[lay]
    cells = _cells(layer, captures, lay, seed=9)
    refs, classes = _refs(layer, runner, cells)
    static_inputs = [(x.clone(), lg.clone()) for _, x, lg in cells]
    main = torch.cuda.current_stream()
    gstream = torch.cuda.Stream()
    gstream.wait_stream(main)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=gstream):
        gouts = [runner(x, lg) for x, lg in static_inputs]
    side = torch.cuda.Stream()
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    small_w = torch.randn(512, fx.HIDDEN, device="cuda", dtype=torch.bfloat16)
    with _Watchdog():
        for rep in range(10):
            side.wait_stream(main)
            with torch.cuda.stream(side):
                for _ in range(3):
                    torch.matmul(a, b)
                    # shared-expert-sized GEMMs as in the served overlap
                    for x, _ in static_inputs[:4]:
                        torch.nn.functional.linear(x, small_w)
            graph.replay()
            torch.cuda.synchronize()
            for i, (o, r) in enumerate(zip(gouts, refs)):
                _identical(o, r, f"gemm co-run rep{rep} #{i}")
    print(f"[routed_moe] co-run GEMM ok; refs vs native {classes}")


# ---------------------------------------------------------------------------
# (d) many back-to-back launches in one graph, several real layers
# ---------------------------------------------------------------------------
def test_graph_many_launches_real_layers(layers, runners, captures):
    """One graph, 12 launches per layer interleaved over the layers (one
    shared workspace, every fourth launch in raw mode, an unrelated consumer
    after each); every replayed output equals the eager routed_moe output for its
    input bit for bit and is classified against native.
    """
    per_layer = 12
    plan = []  # (layer, finalize)
    for i in range(per_layer):
        for lay in LAYERS:
            plan.append((lay, (i % 4) != 3))
    wins = {lay: fx.real_windows(captures[lay], 6) for lay in LAYERS}
    static_inputs = [
        (
            torch.empty(6, fx.HIDDEN, device="cuda", dtype=torch.bfloat16),
            torch.empty(6, fx.NUM_EXPERTS, device="cuda", dtype=torch.float32),
        )
        for _ in plan
    ]

    def fill(rep):
        used = []
        for j, ((lay, _), (xs, ls)) in enumerate(zip(plan, static_inputs)):
            x, lg = wins[lay][(rep * 977 + j * 31) % len(wins[lay])]
            xs.copy_(x)
            ls.copy_(lg)
            used.append((x, lg))
        return used

    used = fill(0)
    main = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        for (lay, fin), (xs, ls) in zip(plan, static_inputs):
            runners[lay](xs, ls, do_finalize=fin)
    main.wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    sink = torch.zeros(6, fx.HIDDEN, device="cuda", dtype=torch.float32)
    with torch.cuda.graph(graph, stream=stream):
        outs = []
        for (lay, fin), (xs, ls) in zip(plan, static_inputs):
            o = runners[lay](xs, ls, do_finalize=fin)
            outs.append(o)
            # an unrelated consumer between launches, as the model's layers
            sink.add_(o.float() if fin else o[0][:6].float())
    classes = {}
    emul_left = 8
    for rep in range(6):
        if rep:
            used = fill(rep)
        graph.replay()
        torch.cuda.synchronize()
        for j, ((lay, fin), o, (x, lg)) in enumerate(zip(plan, outs, used)):
            tag = f"rep{rep} launch{j} layer{lay}"
            e_fin, e_raw = _routed_moe_both(runners[lay], x, lg)
            _identical(o, e_fin if fin else e_raw, tag)
            cls, _ = _classify(layers[lay], x, lg, e_fin, e_raw, tag, emul=emul_left > 0)
            if cls == "budget":
                emul_left -= 1
            classes[cls] = classes.get(cls, 0) + 1
    print(
        f"[routed_moe] graph of {len(plan)} launches over layers {LAYERS}: 6 replays bit-identical "
        f"to eager routed_moe; vs native {classes}"
    )


# ---------------------------------------------------------------------------
# Real routed-MoE inputs (native capture): correctness, not only timing
# ---------------------------------------------------------------------------
EMUL_PER_CASE = int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_EMUL_PER_CASE", "24"))


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_real_windows_match_native(layers, runners, decoders, captures, rows):
    """Every real window of every captured layer: exact expert IDs and
    scales; finalized output exact, FP32-reorder-close, or (sampled, FP64
    emulation) within the frozen budget.
    """
    stride = {1: 7, 6: 6, 8: 8}[rows]
    summary = {}
    for lay in LAYERS:
        layer, runner = layers[lay], runners[lay]
        wins = fx.real_windows(captures[lay], rows, stride)
        assert wins and wins[0][0] is not None, f"layer {lay}: no captured hidden states"
        classes = {}
        nrms = []
        emul_left = EMUL_PER_CASE
        for i, (x, lg) in enumerate(wins):
            fin, raw = _routed_moe_both(runner, x, lg)
            n_fin = layer.native_forward(x, lg)
            n_raw = layer.native_forward(x, lg, do_finalize=False)
            emul_fn = None
            if emul_left > 0:

                def emul_fn(x=x, lg=lg):
                    return fx.emulate_moe(layer, x, lg)

            cls, rep = fx.classify_vs_native(
                fin,
                raw,
                n_fin,
                n_raw,
                emul_fn,
                f"layer {lay} window {i}",
                lambda lg=lg: decoders[lay].compare(runner, lg),
            )
            if cls == "budget":
                emul_left -= 1
            classes[cls] = classes.get(cls, 0) + 1
            if cls != "exact":
                nrms.append(fx.error_stats(fin, n_fin)["nrms"])
        gaps = torch.cat([fx.boundary_gap(lg, layer.bias) for _, lg in wins])
        summary[lay] = {
            "windows": len(wins),
            "classes": classes,
            "max_nrms_non_exact": max(nrms) if nrms else 0.0,
            "rows_exact_boundary_tie": int((gaps == 0).sum()),
            "rows_gap_le_1ulp": int((gaps <= 4.8e-7).sum()),
            "min_gap": float(gaps.min()),
        }
    print(f"[routed_moe] real windows R={rows} rank={RANK}: {summary}")


def test_real_routing_all_layers(layers, decoders, captures):
    """Routing on the real logits of every captured MoE layer with that
    layer's own router bias: the experts of every (t, k), decoded exactly
    from the FC2 rows of a probe activation (``ExpertDecoder``), and the
    expert scale bits equal native's. The expert weights are layer
    ``LAYERS[0]``'s; routing depends only on logits and bias.
    """
    import glob
    import re

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        RoutedMoe,
    )

    base = LAYERS[0]
    layer = layers[base]
    decoder = decoders[base]
    all_layers = sorted(
        {
            int(re.search(r"_layer(\d+)\.pt$", p).group(1))
            for p in glob.glob(os.path.join(fx.REAL_DIR, "rows*_layer*.pt"))
        }
    )
    totals = {"layers": len(all_layers), "windows": 0, "min_margin": float("inf")}
    totals.update({"exact_ties": 0, "gap_le_1ulp": 0, "tie_windows_checked": 0})
    for lay in all_layers:
        cap = [(None, lg.cuda()) for _, lg in fx.load_real_capture(lay)]
        bias = fx.load_router_bias(lay)
        runner = RoutedMoe.from_native(layer.module, bias)
        for i, (_, lg) in enumerate(fx.real_windows(cap, 6)):
            dec = decoder.compare(runner, lg, bias=bias)
            assert dec["ids_equal"] and dec["scale_exact"], (lay, i, dec)
            totals["windows"] += 1
            totals["min_margin"] = min(totals["min_margin"], dec["margin"])
            if bool((fx.boundary_gap(lg, bias) <= 4.8e-7).any()):
                totals["tie_windows_checked"] += 1
        gaps = torch.cat([fx.boundary_gap(lg, bias) for _, lg in cap])
        totals["exact_ties"] += int((gaps == 0).sum())
        totals["gap_le_1ulp"] += int((gaps <= 4.8e-7).sum())
    print(f"[routed_moe] real routing over all layers (decoded experts): {totals}")
    assert totals["min_margin"] > 4, totals
