# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_SHARED_PIN (shared_mxfp8) on an SM107 GPU.

In the served DEP process (the pinned table armed) the shared expert's hand-off GEMMs, gate_up [6144, 6144] and down
[6144, 3072], write the same bits with DEP_SHARED_PIN on and off at 32 / 33 / 44 / 48 / 64 / 65 rows, eager and on a
CUDA graph replay. With it on, 33..64 rows take the pinned CuTe DSL path at the table's entries (both GEMMs; gate_up's
own tactic is not engaged after the full-chain read); 32 rows stay off the table, 65 rows on DEP_GEMM_PROVIDER.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    merge_mxfp8,
    shared_mxfp8,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.gemm import (
    mxfp8_cute_pinned_tactics as pin,
)
from tensorrt_llm._torch.modules import linear as _linear
from tensorrt_llm._torch.modules.linear import MXFP8LinearMethod


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _sm107(), reason="needs an SM107 GPU")

GEMMS = {
    "gate_up": (
        (6144, 6144),
        shared_mxfp8.DepSharedGateUpMxfp8LinearMethod,
        shared_mxfp8.GATE_UP_SLOT,
    ),
    "down": ((6144, 3072), shared_mxfp8.DepSharedDownMxfp8LinearMethod, shared_mxfp8.DOWN_SLOT),
}
ROWS = (32, 33, 44, 48, 64, 65)


@pytest.fixture(autouse=True)
def _served(monkeypatch):
    """The served DEP process: the family table on, DEP_GEMM_PROVIDER on."""
    monkeypatch.setattr(pin, "ENABLED", True)
    monkeypatch.setattr(merge_mxfp8._provider, "DEP_GEMM_PROVIDER", True)
    monkeypatch.setattr(merge_mxfp8, "DEP_PROJ_CUTE", True)


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


@torch.inference_mode()
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("name", list(GEMMS))
def test_on_and_off_write_the_same_bits(name, rows, monkeypatch):
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
    outs = {}
    for on in (False, True):
        monkeypatch.setattr(shared_mxfp8, "DEP_SHARED_PIN", on)
        seen.clear()
        outs[on] = _apply(cls, slot, module, x, a_mx, a_sf)
        torch.cuda.synchronize()
        assert slot.pending is None
        if on and 33 <= rows <= 64:
            expected = pin.TACTICS[(n, k, 64)]
            assert (f"DEP_SHARED_PIN {cls.consumer}", expected) in seen, seen
        if rows == 32:
            assert all(t is None for _, t in seen)
    assert torch.equal(outs[True].view(torch.int16), outs[False].view(torch.int16))


@torch.inference_mode()
@pytest.mark.parametrize("rows", (33, 48, 64))
@pytest.mark.parametrize("name", list(GEMMS))
def test_a_graph_replays_the_engaged_gemm_bit_for_bit(name, rows, monkeypatch):
    (n, k), cls, slot = GEMMS[name]
    a_mx, a_sf, w_mx, w_sf = _operands(n, k, rows, seed=100 + rows)
    module = types.SimpleNamespace(weight=w_mx, weight_scale=w_sf, dtype=torch.bfloat16)
    x = torch.empty(rows, k, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(shared_mxfp8, "DEP_SHARED_PIN", True)
    eager = _apply(cls, slot, module, x, a_mx, a_sf).clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = _apply(cls, slot, module, x, a_mx, a_sf)
    for _ in range(3):
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), eager.view(torch.int16))
