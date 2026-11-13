# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SOFTMAX_MC: the spec sampler's softmax as
``mv2_minimax_m3::minimax_m31_softmax_mc`` through ``ops/sampling/m31_softmax_mc``, installed as the sampler's softmax
override.

It must equal ``flashinfer.sampling.softmax(logits, temperature)`` bit for bit (the microbenchmark generators: randn /
peaked / ties / near_uniform / all_equal / masked / wide rows, temperature 1 / per-row mixed / scalar). Without the
engine's override hook the target installs nothing. TRTLLM_M31_DISABLE_SOFTMAX_MC=1 installs nothing.
"""

import os
import subprocess
import sys

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.sampling import (  # noqa: E501
        m31_softmax_mc as entry,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

TARGET = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2"
CLASSES = ("randn", "peaked", "ties", "near_uniform", "all_equal", "masked", "wide")
TEMPS = ("t1", "mixed", "scalar")
op = entry.m31_softmax_mc


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("SOFTMAX_MC is built for sm_107 here")
    assert entry.softmax_mc_op() is not None, "minimax_m31_softmax_mc is not built"
    pytest.importorskip("flashinfer.sampling")


def _make(cls, rows, d, seed):
    """The microbenchmark logit generators."""
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
    # The eager warm-up on the side stream shares the persistent workspace with the default-stream
    # calls above (and reads their inputs): order it after them.
    stream.wait_stream(torch.cuda.current_stream())
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
    wide = _make("randn", 449, d, 2)
    assert entry.softmax_mc(wide, torch.full((449,), 0.9, device="cuda")) is None
    assert entry.softmax_mc(logits.cpu(), temperature.cpu()) is None


def test_the_target_declines_without_the_override_hook(monkeypatch) -> None:
    """Without the sampler's softmax override hook in the engine the target installs nothing and keeps the stock
    call, even with SOFTMAX_MC on and the op built."""
    _gpu()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        modeling,
    )

    monkeypatch.setattr(modeling, "_SET_SOFTMAX_OVERRIDE", None)
    saved = dict(modeling._SOFTMAX_MC_STATE)
    modeling._SOFTMAX_MC_STATE.clear()
    try:
        modeling._install_softmax_mc()
        assert modeling._SOFTMAX_MC_STATE["installed"] is False
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


def test_write_ctas_follow_the_rows() -> None:
    """16 write CTAs up to 12 rows (every c1 call unchanged), 8 up to 24, 4 up to MAX_ROWS (128)."""
    assert entry.WIDE and entry.max_rows() == entry.MAX_ROWS == 128
    for rows, ctas in (
        (1, 16),
        (8, 16),
        (12, 16),
        (13, 8),
        (24, 8),
        (25, 4),
        (48, 4),
        (96, 4),
        (112, 4),
        (113, 4),
        (128, 4),
    ):
        assert entry.write_ctas(rows) == ctas, rows


@pytest.mark.parametrize("rows", [9, 13, 24, 25, 56, 96, 97, 112, 113, 127, 128])
def test_wide_rows_bitwise_flashinfer(rows) -> None:
    """Between and at the table's edges (the K7 draft's 7c rows included): the override equals FlashInfer's softmax
    bit for bit, with every temperature mode."""
    _gpu()
    for mode in TEMPS:
        for cls in ("randn", "peaked", "masked"):
            logits = _make(cls, rows, 200064, 7 * rows + len(cls))
            temperature = _temperature(mode, rows, rows)
            got = entry.softmax_mc(logits, temperature)
            assert got is not None, "a wide in-contract input must be served"
            _same(got, _fi(logits, temperature), f"{cls} {mode} rows={rows}")


@pytest.mark.parametrize("rows", [48, 113, 128])
def test_wide_graph_replay_equals_eager(rows) -> None:
    """Wide rows replay the eager bits with fresh logits copied in, including the Path-A cap."""
    _gpu()
    logits = _make("peaked", rows, 200064, 8).clone()
    temperature = _temperature("mixed", rows, 8)
    entry.prebuild(logits.device)
    stream = torch.cuda.Stream()
    # The eager warm-up on the side stream shares the persistent workspace with the default-stream
    # calls above (and reads their inputs): order it after them.
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        entry.softmax_mc(logits, temperature)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = entry.softmax_mc(logits, temperature)
    assert out is not None
    for k, cls in enumerate(("randn", "near_uniform")):
        logits.copy_(_make(cls, rows, 200064, 60 + k))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _same(out, _fi(logits, temperature), f"graph replay {cls}")


def test_the_wide_kill_switch_restores_the_c1_form() -> None:
    """The wide kill restores eight rows; the 128-row item kill restores 112. Log each cap once."""
    _gpu()
    probe = (
        f"import importlib, torch; m = importlib.import_module('{TARGET}.ops.sampling.m31_softmax_mc'); "
        "m.softmax_mc_op(); x = torch.randn(9, 200064, device='cuda'); "
        "y = torch.randn(128, 200064, device='cuda'); "
        "print('CAP', m.max_rows(), m.write_ctas(40), m.softmax_mc(x, None) is None, "
        "m.softmax_mc(y, None) is None)"
    )
    for flag, cap_flag, want, line in (
        ("1", "0", "CAP 8 16 True True", "<= 8, write CTAs 16"),
        ("0", "0", "CAP 128 4 False False", "<= 128,"),
        ("0", "1", "CAP 112 4 False True", "<= 112,"),
    ):
        out = subprocess.run(
            [sys.executable, "-c", probe],
            env=dict(
                os.environ,
                TLLM_LOG_LEVEL="INFO",
                TRTLLM_M31_DISABLE_SOFTMAX_MC_WIDE=flag,
                TRTLLM_M31_DISABLE_SOFTMAX_MC_128=cap_flag,
            ),
            capture_output=True,
            text=True,
            check=True,
        )
        assert [ln for ln in out.stdout.splitlines() if ln.startswith("CAP ")][-1] == want
        lines = [ln for ln in (out.stdout + out.stderr).splitlines() if "SOFTMAX_MC rows:" in ln]
        assert len(lines) == 1 and line in lines[0], lines
