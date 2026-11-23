# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SOFTMAX_MC: the spec sampler's softmax as ``mv2_minimax_m3::minimax_m31_softmax_mc`` through
``ops/sampling/m31_softmax_mc``.

It must equal ``flashinfer.sampling.softmax(logits, temperature)`` bit for bit (peaked / ties / masked / wide rows,
temperature 1 / per-row mixed / scalar), and no temperature is temperature 1."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.sampling import (  # noqa: E501
    m31_softmax_mc as entry,
)

CLASSES = ("peaked", "ties", "masked", "wide")
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
