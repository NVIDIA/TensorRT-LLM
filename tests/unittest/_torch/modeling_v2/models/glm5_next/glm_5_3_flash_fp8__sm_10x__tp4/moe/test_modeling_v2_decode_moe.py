# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm5_next fp8 tp4 decode_moe entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.moe.decode_moe import (  # noqa: E501
    decode_moe as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Both sides dequantize the same FP8 weights exactly and keep the activations in FP32 up to
    the single BF16 rounding of the output; they differ in the order of the FP32 sums over 4096
    and 512 products (FMA tiles against torch's matmul), ~1e-6 of the row scale, plus the
    reference's FP32 SiLU against the kernel's exp-based one, ~1e-7 relative. 4 ulp element-wise
    and 2 ulp RMS admit rare one-ulp flips; a wrong expert, slot weight or clamp is off by the
    size of an expert's contribution, many ulp."""
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _build(spec: dict, seed: int):
    """One rank's experts at moe_tp4 shapes, with scales and weights giving O(1) outputs.

    Logit gaps far above FP32 noise keep the top-8 unambiguous, so the reference's
    ``torch.topk`` and the kernel's tie rule pick the same experts.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, e, h, i = spec["m"], spec["e"], spec["h"], spec["i"]

    def fp8(*shape):
        return (torch.randn(*shape, generator=g, device="cuda") * 0.25).to(torch.float8_e4m3fn)

    def scales(*shape, fan_in):
        return torch.rand(*shape, generator=g, device="cuda") * fan_in**-0.5 + 0.5 * fan_in**-0.5

    x = torch.randn(m, h, generator=g, device="cuda").to(torch.bfloat16)
    logits = torch.randperm(e, generator=g, device="cuda").float().repeat(m, 1) * 0.01
    logits += torch.arange(m, device="cuda")[:, None] * 0.37
    bias = torch.zeros(e, device="cuda")
    w31, s31 = fp8(e, 2 * i, h), scales(e, 2 * i // 128, h // 128, fan_in=h)
    w2, s2 = fp8(e, h, i), scales(e, h // 128, i // 128, fan_in=i)
    sw31, ss31 = fp8(2 * i, h), scales(2 * i // 128, h // 128, fan_in=h)
    sw2, ss2 = fp8(h, i), scales(h // 128, i // 128, fan_in=i)
    return (x, logits, bias, 2.5, 10.0, w31, s31, w2, s2, sw31, ss31, sw2, ss2)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    _compare(op(*args), op.reference(*args))


def test_clamp_is_applied() -> None:
    """A tiny SwiGLU limit makes the clamped and unclamped products disagree; the op must match
    the clamped reference."""
    args = list(_build(op.CELLS[1].spec, seed=7))
    args[4] = 0.05
    _compare(op(*args), op.reference(*args))


def test_independent_rows() -> None:
    """Rows routed on their own logits with a nonzero correction bias: the step reads ~30 distinct
    experts (the kernels stream each once), and the bias picks the experts but not their weights."""
    spec = op.CELLS[1].spec
    args = list(_build(spec, seed=13))
    g = torch.Generator(device="cuda").manual_seed(14)
    args[1] = torch.randn(spec["m"], spec["e"], generator=g, device="cuda")
    args[2] = (torch.rand(spec["e"], generator=g, device="cuda") - 0.5) * 0.1
    _compare(op(*args), op.reference(*args))


def test_cuda_graph_capture_and_replay() -> None:
    spec = op.CELLS[1].spec
    args = _build(spec, seed=31)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op(*args)
    fresh = _build(spec, seed=32)
    args[0].copy_(fresh[0])
    args[1].copy_(fresh[1])
    graph.replay()
    torch.cuda.synchronize()
    _compare(out, op.reference(*args))
