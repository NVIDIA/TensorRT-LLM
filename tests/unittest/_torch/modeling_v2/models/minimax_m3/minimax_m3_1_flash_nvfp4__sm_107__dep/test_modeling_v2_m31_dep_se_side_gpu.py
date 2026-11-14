# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SE_SIDE (se_side) on a 212-SM SM107 GPU.

* The green partition splits the device into 184 / 28 SMs, %smid-probed disjoint, and is cached.
* In the served DEP process (the pinned table armed, DEP_GEMM_PROVIDER on) the shared expert's hand-off GEMMs, gate_up
  [6144, 6144] and down [6144, 3072], write the same bits on the 28-SM side stream (inside ``se_side.side_calls``) as
  on the main stream, at 33 / 44 / 48 / 64 / 88 / 128 rows, and a CUDA graph that forks them onto the side stream and
  joins back replays them bit for bit.
* Co-residency soak: a 184-CTA grid barrier (one CTA per SM) on the main stream, the stand-in for the persistent routed
  MegaMoE grid, always completes beside the shared expert's GEMMs on the side stream (no CTA times out in N rounds);
  its negative control, one CTA more than the device holds, times out, so the check can fail.
"""

import ctypes
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    merge_mxfp8,
    se_side,
    shared_mxfp8,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import green_partition
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as pin,
)
from tensorrt_llm._torch.modules import linear as _linear
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod


def _sm107_212() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 7)
        and torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        == se_side.NUM_SMS
    )


pytestmark = pytest.mark.skipif(not _sm107_212(), reason="needs a 212-SM SM107 GPU")

GEMMS = {
    "gate_up": (
        (6144, 6144),
        shared_mxfp8.DepSharedGateUpMxfp8LinearMethod,
        shared_mxfp8.GATE_UP_SLOT,
    ),
    "down": ((6144, 3072), shared_mxfp8.DepSharedDownMxfp8LinearMethod, shared_mxfp8.DOWN_SLOT),
}
ROWS = (33, 44, 48, 64, 88, 128)
SOAK_ROUNDS = 200

_BARRIER_SRC = r"""
extern "C" __global__ void grid_barrier(int* counter, int* timed_out, int target, long long spin_limit) {
    extern __shared__ int smem[];
    if (threadIdx.x == 0) {
        smem[0] = 1;
        atomicAdd(counter, 1);
        long long spins = 0;
        while (*((volatile int*)counter) < target) {
            if (++spins > spin_limit) {
                atomicExch(timed_out, 1);
                break;
            }
        }
    }
}
"""


@pytest.fixture(autouse=True)
def _served(monkeypatch):
    """The served DEP process: the family table on, DEP_GEMM_PROVIDER on."""
    monkeypatch.setattr(pin, "ENABLED", True)
    monkeypatch.setattr(merge_mxfp8._provider, "DEP_GEMM_PROVIDER", True)
    monkeypatch.setattr(merge_mxfp8, "DEP_PROJ_CUTE", True)
    monkeypatch.setattr(shared_mxfp8, "DEP_SHARED_PIN", True)


@pytest.fixture(scope="module")
def part():
    return green_partition.get(torch.cuda.current_device(), se_side.MAIN_SMS)


def _operands(n, k, rows, seed):
    from flashinfer import mxfp8_quantize

    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.02
    a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16, generator=gen)
    w_mx, w_sf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
    a_mx, a_sf = mxfp8_quantize(a, is_sf_swizzled_layout=True)
    return a_mx, a_sf, w_mx, w_sf


def _method(cls):
    method = MXFP8LinearMethod()
    method.__class__ = cls
    method.use_cutlass = True
    method.backend = "trtllm"
    method.use_native_autotuner = False
    method._flashinfer_autotuned = False
    method.tune_decode_graph_backends = True
    return method


def _apply(cls, slot, module, x, a_mx, a_sf):
    slot.publish(x, a_mx, a_sf)
    token = _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.set(True)
    try:
        return _method(cls).apply(module, x, None)
    finally:
        _linear._FLASHINFER_MXFP8_DECODE_GRAPH_CAPTURE_ACTIVE.reset(token)


def _on_side(part, fn):
    """``fn()`` forked onto the side stream inside side_calls, joined back to the current stream."""
    cur = torch.cuda.current_stream()
    part.side_stream.wait_stream(cur)
    with torch.cuda.stream(part.side_stream), se_side.side_calls():
        out = fn()
    cur.wait_stream(part.side_stream)
    return out


def test_the_partition_is_184_and_28_disjoint_and_cached(part):
    assert part.num_sms == se_side.NUM_SMS
    assert (
        len(part.main_sms) == se_side.MAIN_SMS
        and len(part.side_sms) == se_side.NUM_SMS - se_side.MAIN_SMS
    )
    assert part.disjoint and (part.main_sms | part.side_sms) == frozenset(range(part.num_sms))
    assert green_partition.get(torch.cuda.current_device(), se_side.MAIN_SMS) is part


@torch.inference_mode()
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name", list(GEMMS))
def test_the_side_stream_writes_the_main_streams_bits(part, name, rows, monkeypatch):
    (n, k), cls, slot = GEMMS[name]
    a_mx, a_sf, w_mx, w_sf = _operands(n, k, rows, seed=rows + n % 7)
    module = types.SimpleNamespace(weight=w_mx, weight_scale=w_sf, dtype=torch.bfloat16)
    x = torch.empty(rows, k, device="cuda", dtype=torch.bfloat16)
    seen = []
    real = pin.tactic_for

    def tactic_for(m, nn, kk):
        t = real(m, nn, kk)
        seen.append((pin._CALLER.get(), t))
        return t

    monkeypatch.setattr(pin, "tactic_for", tactic_for)
    main = _apply(cls, slot, module, x, a_mx, a_sf)
    torch.cuda.synchronize()
    seen.clear()
    side = _on_side(part, lambda: _apply(cls, slot, module, x, a_mx, a_sf))
    torch.cuda.synchronize()
    assert slot.pending is None
    assert torch.equal(side.view(torch.int16), main.view(torch.int16))
    if rows <= 64:
        assert (f"DEP_SHARED_PIN {cls.consumer} (DEP_SE_SIDE)", pin.TACTICS[(n, k, 64)]) in seen, (
            seen
        )


@torch.inference_mode()
@pytest.mark.parametrize("rows", (44, 88, 128))
def test_a_graph_forks_the_shared_gemms_onto_the_side_and_replays_them(part, rows):
    ops = []
    for name in ("gate_up", "down"):
        (n, k), cls, slot = GEMMS[name]
        a_mx, a_sf, w_mx, w_sf = _operands(n, k, rows, seed=300 + rows + n % 5)
        module = types.SimpleNamespace(weight=w_mx, weight_scale=w_sf, dtype=torch.bfloat16)
        x = torch.empty(rows, k, device="cuda", dtype=torch.bfloat16)
        ops.append((cls, slot, module, x, a_mx, a_sf))

    def both():
        return [_apply(*op) for op in ops]

    eager = [o.clone() for o in _on_side(part, both)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        outs = _on_side(part, both)
    for _ in range(3):
        for o in outs:
            o.zero_()
        graph.replay()
        torch.cuda.synchronize()
        for o, e in zip(outs, eager):
            assert torch.equal(o.view(torch.int16), e.view(torch.int16))


class _Barrier:
    """A grid barrier: every CTA (one per SM, by its shared memory) arrives, then waits for all. A CTA that waits
    past ``spin_limit`` polls sets ``timed_out`` and leaves: a grid that cannot be co-resident fails, not hangs."""

    SMEM = 150 * 1024

    def __init__(self):
        from cuda.bindings import driver as d
        from cuda.bindings import nvrtc

        self.d = d
        major, minor = torch.cuda.get_device_capability()
        prog = green_partition._ck(
            nvrtc.nvrtcCreateProgram(_BARRIER_SRC.encode(), b"grid_barrier.cu", 0, [], []), "nvrtc"
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        assert int(nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]) == 0
        size = green_partition._ck(nvrtc.nvrtcGetCUBINSize(prog), "cubin size")
        cubin = b" " * size
        green_partition._ck(nvrtc.nvrtcGetCUBIN(prog, cubin), "cubin")
        torch.cuda.init()
        self.mod = green_partition._ck(d.cuModuleLoadData(cubin), "module")
        self.fn = green_partition._ck(d.cuModuleGetFunction(self.mod, b"grid_barrier"), "function")
        green_partition._ck(
            d.cuFuncSetAttribute(
                self.fn,
                d.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                self.SMEM,
            ),
            "smem attribute",
        )
        self.counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.timed_out = torch.zeros(1, dtype=torch.int32, device="cuda")

    def launch(self, ctas, spin_limit):
        self.counter.zero_()
        args = (
            (
                ctypes.c_void_p(self.counter.data_ptr()),
                ctypes.c_void_p(self.timed_out.data_ptr()),
                ctypes.c_int(ctas),
                ctypes.c_longlong(spin_limit),
            ),
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong),
        )
        stream = self.d.CUstream(torch.cuda.current_stream().cuda_stream)
        green_partition._ck(
            self.d.cuLaunchKernel(self.fn, ctas, 1, 1, 32, 1, 1, self.SMEM, stream, args, 0),
            "barrier launch",
        )


@torch.inference_mode()
def test_the_routed_grid_stays_co_resident_beside_the_side_stream(part):
    barrier = _Barrier()
    # Negative control: one CTA more than the device holds at one per SM can never all arrive.
    barrier.timed_out.zero_()
    barrier.launch(part.num_sms + 1, 1 << 20)
    torch.cuda.synchronize()
    assert int(barrier.timed_out.item()) == 1
    rows = 128
    ops = []
    for name in ("gate_up", "down"):
        (n, k), cls, slot = GEMMS[name]
        a_mx, a_sf, w_mx, w_sf = _operands(n, k, rows, seed=500 + n % 3)
        module = types.SimpleNamespace(weight=w_mx, weight_scale=w_sf, dtype=torch.bfloat16)
        x = torch.empty(rows, k, device="cuda", dtype=torch.bfloat16)
        ops.append((cls, slot, module, x, a_mx, a_sf))
    barrier.timed_out.zero_()
    for i in range(SOAK_ROUNDS):
        cur = torch.cuda.current_stream()
        part.side_stream.wait_stream(cur)
        with torch.cuda.stream(part.side_stream), se_side.side_calls():
            for _ in range(4):
                for op in ops:
                    _apply(*op)
        if i % 2:
            torch.cuda._sleep(2000)  # let the side GEMMs take their SMs first on odd rounds
        barrier.launch(se_side.MAIN_SMS, 1 << 24)
        cur.wait_stream(part.side_stream)
    torch.cuda.synchronize()
    assert int(barrier.timed_out.item()) == 0
