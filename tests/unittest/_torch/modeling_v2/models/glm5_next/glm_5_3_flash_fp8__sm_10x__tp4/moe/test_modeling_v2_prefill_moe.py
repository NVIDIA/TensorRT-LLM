# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm5_next fp8 tp4 prefill_moe entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.moe.prefill_moe import (  # noqa: E501
    PrefillMoeWorkspace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.moe.prefill_moe import (  # noqa: E501
    prefill_moe as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Both sides take the same FP8 inputs and weights, round the FC1 output to BF16 and
    requantize the intermediate 1 x 128 to FP8. They differ in the order of the FP32 sums of
    the GEMMs, so a BF16 FC1 output can round the other way and move one FP8 code of the
    intermediate: a relative error far below 1e-3 overall. A wrong expert, slot weight, clamp
    or scale is off by an expert's whole contribution, O(10%)."""
    err = (out.float() - ref.float()).norm() / ref.float().norm()
    assert err < 1e-3, f"relative L2 error {err:.2e}"
    assert torch.isfinite(out).all()


def _build(spec: dict, seed: int):
    """One rank's experts at moe_tp4 shapes and FP8 rows with 1 x 128 scales, O(1) outputs.

    Logit gaps far above FP32 noise keep the top-8 unambiguous, so the reference's
    ``torch.topk`` and the kernel's tie rule pick the same experts. The addend (the shared
    expert's output in the model) is scaled to 0.1, so the routed experts, not the addend,
    dominate the output the comparison measures.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, e, h, i = spec["t"], spec["e"], spec["h"], spec["i"]

    def fp8(*shape):
        return (torch.randn(*shape, generator=g, device="cuda") * 0.25).to(torch.float8_e4m3fn)

    def scales(*shape, fan_in):
        return torch.rand(*shape, generator=g, device="cuda") * fan_in**-0.5 + 0.5 * fan_in**-0.5

    x = torch.randn(t, h, generator=g, device="cuda").to(torch.bfloat16)
    x_fp8, x_scale = torch.ops.trtllm.fp8_quantize_1x128(x)
    logits = torch.rand(t, e, generator=g, device="cuda") * 4.0
    bias = torch.randn(e, generator=g, device="cuda") * 0.1
    w31, s31 = fp8(e, 2 * i, h), scales(e, 2 * i // 128, h // 128, fan_in=h)
    w2, s2 = fp8(e, h, i), scales(e, h // 128, i // 128, fan_in=i)
    addend = (torch.randn(t, h, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    return (
        x_fp8,
        x_scale,
        logits,
        bias,
        2.5,
        10.0,
        w31,
        s31,
        w2,
        s2,
        addend,
        PrefillMoeWorkspace(),
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    op.is_valid(*args)
    _compare(op(*args), op.reference(*args))


def test_clamp_is_applied() -> None:
    """A tiny SwiGLU limit makes the clamped and unclamped products disagree; the op must match
    the clamped reference."""
    args = list(_build(op.CELLS[1].spec, seed=7))
    args[5] = 0.05
    _compare(op(*args), op.reference(*args))


def test_skewed_routing() -> None:
    """Most rows on a few experts: segments of many tiles next to empty experts."""
    args = list(_build(op.CELLS[1].spec, seed=11))
    args[2][:, :4] += 100.0
    _compare(op(*args), op.reference(*args))
