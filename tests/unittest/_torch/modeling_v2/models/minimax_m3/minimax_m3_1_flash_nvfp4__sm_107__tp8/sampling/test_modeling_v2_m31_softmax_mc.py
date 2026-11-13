# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SOFTMAX_MC: the spec sampler's softmax as
``mv2_minimax_m3::minimax_m31_softmax_mc`` through ``ops/sampling/m31_softmax_mc``, installed as the sampler's softmax
override where the engine offers one.

It must equal ``flashinfer.sampling.softmax(logits, temperature)`` bit for bit (the microbench generators: randn /
peaked / ties / near_uniform / all_equal / masked / wide rows, temperature 1 / per-row mixed / scalar). Outside its
contract the override declines. The target installs it only through the engine's override setter;
TRTLLM_M31_DISABLE_SOFTMAX_MC=1 installs nothing.
"""

import os
import subprocess
import sys
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.sampling import (  # noqa: E501
    m31_softmax_mc as entry,
)

TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8"
CLASSES = ("randn", "peaked", "ties", "near_uniform", "all_equal", "masked", "wide")
TEMPS = ("t1", "mixed", "scalar")
op = entry.m31_softmax_mc


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SOFTMAX_MC is built for sm_107 here")
    assert entry.softmax_mc_op() is not None, "this build has no minimax_m31_softmax_mc"
    pytest.importorskip("flashinfer.sampling")


def _make(cls, rows, d, seed):
    """The microbench logit generators."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if cls == "randn":
        return (torch.randn(rows, d, generator=g, device="cuda") * 3).contiguous()
    if cls == "peaked":  # few dominant tokens, long tail (real-like rows)
        x = torch.randn(rows, d, generator=g, device="cuda") * 2
        idx = torch.randint(0, d, (rows, 8), generator=g, device="cuda")
        x.scatter_(1, idx, 14 + 3 * torch.rand(rows, 8, generator=g, device="cuda"))
        return x.contiguous()
    if cls == "ties":
        return (torch.round(torch.randn(rows, d, generator=g, device="cuda") * 2) / 2).contiguous()
    if cls == "near_uniform":
        return (torch.randn(rows, d, generator=g, device="cuda") * 1e-3).contiguous()
    if cls == "all_equal":
        return torch.zeros(rows, d, device="cuda")
    if cls == "masked":
        x = torch.randn(rows, d, generator=g, device="cuda") * 3
        x[:, d // 2 :] = -1e4
        return x.contiguous()
    if cls == "wide":
        return (torch.randn(rows, d, generator=g, device="cuda") * 30).contiguous()
    raise ValueError(cls)


def _temperature(mode, rows, seed):
    """FlashInfer's temperature argument: an fp32 [rows] tensor, or a Python float."""
    g = torch.Generator(device="cuda").manual_seed(seed + 3)
    if mode == "t1":
        return torch.ones(rows, device="cuda")
    if mode == "mixed":
        return (0.3 + 1.2 * torch.rand(rows, generator=g, device="cuda")).contiguous()
    if mode == "scalar":
        return 0.7
    raise ValueError(mode)


def _fi(logits, temperature):
    import flashinfer.sampling

    return flashinfer.sampling.softmax(logits, temperature)


def _same(a, b, what):
    """Bit for bit."""
    assert torch.equal(a.view(torch.int32), b.view(torch.int32)), f"{what}: fp32 bits differ"


@pytest.mark.parametrize("mode", TEMPS)
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_bitwise_flashinfer_softmax(cell, cls, mode) -> None:
    """At each certified (rows, d): the override equals FlashInfer's softmax bit for bit, and the explicit entry call
    at write_ctas 4 / 8 gives the same bits again."""
    _gpu()
    rows, d = cell.spec["rows"], cell.spec["d"]
    if rows > entry.max_rows():
        pytest.skip(f"{rows} rows: TP8WIDE_SOFTMAX is off (cap {entry.max_rows()})")
    for seed in range(2):
        logits = _make(cls, rows, d, 100 * seed + rows)
        temperature = _temperature(mode, rows, seed)
        got = entry.softmax_mc(logits, temperature)
        assert got is not None, "an in-contract input must be served"
        _same(got, _fi(logits, temperature), f"{cls} {mode} rows={rows} seed={seed}")
        for write_ctas in (4, 8):
            again = op(logits, temperature, entry.prebuild(logits.device), write_ctas)
            _same(again, got, f"write_ctas={write_ctas}")


def test_no_temperature_is_one() -> None:
    """A None temperature (softmax_op's Optional) is FlashInfer's default, 1.0."""
    _gpu()
    logits = _make("peaked", 8, 200064, 11)
    got = entry.softmax_mc(logits, None)
    assert got is not None
    _same(got, _fi(logits, None), "temperature None")


def test_graph_replay_equals_eager() -> None:
    """Captured in a CUDA graph (workspace prebuilt outside the capture), the replay gives the eager bits."""
    _gpu()
    logits = _make("peaked", 8, 200064, 5)
    temperature = _temperature("mixed", 8, 5)
    entry.prebuild(logits.device)
    ref = entry.softmax_mc(logits, temperature)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        entry.softmax_mc(logits, temperature)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = entry.softmax_mc(logits, temperature)
    assert out is not None, "a prebuilt workspace must serve a captured call"
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _same(out, ref, "graph replay")


def test_outside_the_contract_the_override_declines() -> None:
    """A non-fp32 / misaligned / non-contiguous logits, a d % 4 width, a vocab below FlashInfer's Path A, a mis-sized,
    non-fp32 or 2-D temperature, a non-number temperature or more rows than MAX_ROWS return None (the sampler runs
    stock FlashInfer)."""
    _gpu()
    d = 200064
    logits = _make("randn", 4, d, 1)
    temperature = torch.full((4,), 0.9, device="cuda")
    assert entry.softmax_mc(logits.half(), temperature) is None
    assert entry.softmax_mc(logits, temperature.double()) is None
    assert entry.softmax_mc(logits, temperature[:3]) is None
    assert entry.softmax_mc(logits, temperature.view(4, 1)) is None
    assert entry.softmax_mc(logits, "0.9") is None
    assert entry.softmax_mc(logits[:, : d - 4], temperature) is None
    assert entry.softmax_mc(logits[:, : d - 2].contiguous(), temperature) is None
    assert (
        entry.softmax_mc(logits[:, : entry.PATH_A_MIN_VOCAB - 4].contiguous(), temperature) is None
    )
    flat = torch.empty(4 * d + 1, device="cuda")
    assert entry.softmax_mc(flat[1:].view(4, d), temperature) is None
    cap = entry.max_rows()  # TP8WIDE_SOFTMAX: 32, else 8
    wide = _make("randn", cap + 1, d, 2)
    assert entry.softmax_mc(wide, torch.full((cap + 1,), 0.9, device="cuda")) is None
    assert entry.softmax_mc(logits.cpu(), temperature.cpu()) is None


def test_the_target_installs_the_override(monkeypatch) -> None:
    """With SOFTMAX_MC on (the default) and the op built, the target installs ``softmax_mc`` once per process through
    the
    engine's override setter, and installs nothing on an engine that has no such setter."""
    _gpu()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        modeling,
    )

    if not modeling.SOFTMAX_MC:
        pytest.skip("SOFTMAX_MC is switched off in this environment")
    saved = dict(modeling._SOFTMAX_MC_STATE)
    try:
        for setter in (None, mock.Mock()):
            monkeypatch.setattr(modeling, "_SET_SOFTMAX_OVERRIDE", setter)
            modeling._SOFTMAX_MC_STATE.clear()
            modeling._install_softmax_mc()
            assert modeling._SOFTMAX_MC_STATE["installed"] is (setter is not None)
            if setter is not None:
                setter.assert_called_once_with(entry.softmax_mc)
    finally:
        modeling._SOFTMAX_MC_STATE.clear()
        modeling._SOFTMAX_MC_STATE.update(saved)


def test_the_kill_switch_installs_nothing() -> None:
    """TRTLLM_M31_DISABLE_SOFTMAX_MC=1 (read at import): SOFTMAX_MC is off and the target installs nothing."""
    probe = (
        f"import {TARGET}.modeling as m\n"
        "m._install_softmax_mc()\n"
        "print(m.SOFTMAX_MC, m._SOFTMAX_MC_STATE['installed'])"
    )
    env = dict(os.environ, TRTLLM_M31_DISABLE_SOFTMAX_MC="1")
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "False False"
