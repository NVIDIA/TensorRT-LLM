# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the e4m3_quantize_pow2_dynamic catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and
`compare` are the gate. This file turns a cell's spec into tensors and keeps
the checks a cell cannot carry: CUDA-graph replay on new data, the round-trip
error bound, the wrong-variant controls and the op's own rejections.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.quantization.e4m3_quantize_pow2_dynamic import (  # noqa: E501
    e4m3_quantize_pow2_dynamic as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "e4m3_quantize_pow2_dynamic requires a CUDA device"

DEV = "cuda"
E4M3_MAX = 448.0
FLOOR = 1e-4


def _query_like(T, H, seed, spread=6):
    """bf16 [T, H, 512] with per-(token, head) magnitudes over 2^-spread .. 2^spread (the golden
    queries' |q| reaches 18..88 per layer) and a few exact zeros."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    mag = torch.randint(-spread, spread + 1, (T, H, 1), device=DEV, generator=g).float()
    x = torch.randn(T, H, 512, device=DEV, generator=g) * torch.exp2(mag)
    x[:, :, 7] = 0.0
    return x.to(torch.bfloat16)


def _build(spec, seed):
    if spec["kind"] == "query":
        x = _query_like(spec["T"], spec["H"], seed).to(spec["dtype"])
    else:
        amax = spec["amax"]
        x = torch.zeros(4, 16, 512, dtype=spec["dtype"], device=DEV)
        x[1, 3, 5] = amax
        x[2, 0, 0] = -amax / 3
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=DEV)
    scale_out = torch.full((1,), -1.0, dtype=torch.float32, device=DEV)
    return x, out, scale_out, spec["scale_mult"]


def _scale_of(x):
    """The power-of-two scale `s` the op is specified to pick for `x`."""
    _, ref_scale = op.reference(x, x, torch.zeros(1, device=DEV), 1.0)
    return ref_scale[0].item()


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, out, scale_out, scale_mult = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    result = op(x, out, scale_out, scale_mult)
    assert result[0] is out and result[1] is scale_out, "the op writes in place and returns both"
    op.compare(result, op.reference(x, out, scale_out, scale_mult))

    s = _scale_of(x)
    if cell.spec["kind"] == "query":
        # No value saturates: the largest |x| / s is in [224, 448].
        top = x.float().abs().max().item() / s
        assert 224.0 <= top <= 448.0, top
    elif cell.spec["amax"] == 449.0:
        assert s == 2.0
    elif cell.spec["amax"] <= FLOOR:
        assert s == 2.0**-22  # 2^ceil(log2(1e-4 / 448))


def test_cuda_graph_capture_replay_follows_the_data():
    """The scale is recomputed on the device at every replay (no host sync, no stale amax)."""
    x = _query_like(96, 16, seed=11)
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=DEV)
    so = torch.empty(1, dtype=torch.float32, device=DEV)
    op(x, out, so, 0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(x, out, so, 0.5)
    for seed, spread in ((12, 2), (13, 9), (14, 0)):
        x.copy_(_query_like(96, 16, seed=seed, spread=spread))
        graph.replay()
        op.compare((out, so), op.reference(x, out, so, 0.5))


def test_round_trip_error_is_e4m3_rounding():
    """out * s reproduces x to E4M3's 3-bit mantissa: relative error <= 2^-4 in the normal range."""
    x = _query_like(512, 16, seed=21)
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=DEV)
    so = torch.empty(1, dtype=torch.float32, device=DEV)
    op(x, out, so, 1.0)
    s = so.item()
    back = out.float() * s
    xf = x.float()
    normal = xf.abs() >= 2.0**-6 * s
    rel = ((back - xf).abs() / xf.abs().clamp_min(1e-30))[normal]
    assert rel.max().item() <= 2.0**-4 + 1e-7


def test_wrong_variants_are_detected():
    """Controls: a non-power-of-two scale (amax / 448), and a scale rounded down instead of up."""
    x = _query_like(64, 16, seed=31)
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=DEV)
    so = torch.empty(1, dtype=torch.float32, device=DEV)
    op(x, out, so, 1.0)
    amax = x.float().abs().max().item()
    for name, s in (
        ("amax/448", amax / 448.0),
        ("floor pow2", 2.0 ** torch.floor(torch.log2(torch.tensor(amax / 448.0))).item()),
    ):
        q = (x.float() / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
        frac = (q != out.view(torch.uint8)).float().mean().item()
        assert frac > 0.3, f"control '{name}' differs on only {frac:.3f}"


def test_rejects_bad_inputs():
    x = _query_like(4, 16, seed=41)
    so = torch.empty(1, dtype=torch.float32, device=DEV)
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        op(x, torch.empty_like(x), so, 1.0)
    with pytest.raises(ValueError, match="float32"):
        op(
            x,
            torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=DEV),
            torch.empty(1, dtype=torch.bfloat16, device=DEV),
            1.0,
        )
