# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's helix_post_process_native entry: every cell against the FP64
combine, and the zero-KV sources carrying no weight."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_ull_helix_post_process_native import (  # noqa: E501
    helix_post_process_native as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="certified on SM10x",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)
VALUE = 128


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, sources, heads = spec["tokens"], spec["cp_size"], spec["heads"]
    values = torch.randn(tokens, sources, heads, VALUE, generator=g, device="cuda")
    values = values.to(torch.bfloat16)
    peak = torch.randn(tokens, sources, heads, generator=g, device="cuda") * 4
    total = torch.rand(tokens, sources, heads, generator=g, device="cuda") * 50 + 1
    if spec["masked"]:
        # Every token keeps source 0; the others drop out at random.
        dropped = torch.rand(tokens, sources, 1, generator=g, device="cuda") < 0.4
        dropped[:, 0] = False
        peak = peak.masked_fill(dropped, float("-inf"))
        total = total.masked_fill(dropped, 0.0)
    stats = torch.stack((peak, total), dim=-1).contiguous()
    return values.contiguous(), stats


def _ulp(x: torch.Tensor) -> torch.Tensor:
    mag = x.float().abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(mag)) - 7)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    values, stats = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        out = op(values, stats, 1.0, 1)
    ref = op.reference(values, stats, 1.0, 1)
    assert out.shape == ref.shape == (values.shape[0], values.shape[2] * VALUE)
    assert out.dtype == torch.bfloat16
    # FP32 weights and accumulation over at most 16 sources, one BF16 rounding: within two BF16
    # ulp of the FP64 value (measured below one); a dropped or doubled source moves it by tens.
    err = (out.float() - ref.float()).abs()
    scale = values.float().abs().amax(dim=1).reshape(out.shape)
    assert bool((err <= 2 * _ulp(scale)).all()), f"max err {err.max().item():.3e}"


def test_the_gate_discriminates_a_dropped_source() -> None:
    spec = dict(tokens=8, cp_size=16, heads=6, masked=False)
    values, stats = _build(spec, seed=5)
    out = op(values, stats, 1.0, 1)
    short = op.reference(values[:, 1:].contiguous(), stats[:, 1:].contiguous(), 1.0, 1)
    err = (out.float() - short.float()).abs()
    scale = values.float().abs().amax(dim=1).reshape(out.shape)
    assert not bool((err <= 2 * _ulp(scale)).all()), "the band accepted a combine missing a source"
