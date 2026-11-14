# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm5_next fp8 tp4 prefill_moe entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.moe.prefill_moe import (  # noqa: E501
    PrefillMoeWorkspace,
    route,
    route_reference,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.moe.prefill_moe import (  # noqa: E501
    prefill_moe as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

#: GLM-5.3-Flash at moe_tp4.
_GLM = dict(e=288, h=4096, i=512)

#: The routing weights: the kernel's sigmoid (``0.5 * tanh(0.5 * x) + 0.5`` with libdevice's
#: tanhf) and divisions are within a few FP32 ulp of torch's, and a weight is a ratio of their
#: sums: well under 1e-5 relative. A wrong expert, normalization or scaling factor is off by O(1).
_WEIGHT_RTOL = 1e-5


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Both sides take the same FP8 inputs and weights, round the FC1 output to BF16 and
    requantize the intermediate 1 x 128 to FP8. They differ in the order of the FP32 sums of
    the GEMMs (about 1e-7 relative), so a BF16 FC1 output rounds the other way at a rate of
    about 1e-4 and then moves an FP8 code of the intermediate one step (12.5% of it) about
    once in 30: a relative L2 error of a few 1e-4 on the routed output. A wrong expert, slot
    weight, clamp or scale is off by an expert's whole contribution, O(10%) of a row."""
    err = (out.float() - ref.float()).norm() / ref.float().norm()
    assert err < 1e-3, f"relative L2 error {err:.2e}"
    assert torch.isfinite(out).all()


def _build(spec: dict, seed: int) -> list:
    """One rank's experts at moe_tp4 shapes and FP8 rows with 1 x 128 scales, O(1) outputs.

    The routing choices ``sigmoid(logit) + bias`` are a per-row permutation of a grid 1.4e-3
    apart, so the top-8 and their order are the same however the sigmoid rounds: the
    reference's sort and the kernel's selection pick the same experts in the same slots.
    The addend is the shared expert's output in the model.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, e, h, i = spec["t"], spec["e"], spec["h"], spec["i"]

    def fp8(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=g, device="cuda") * 0.25).to(torch.float8_e4m3fn)

    def scales(*shape: int, fan_in: int) -> torch.Tensor:
        return torch.rand(*shape, generator=g, device="cuda") * fan_in**-0.5 + 0.5 * fan_in**-0.5

    x = torch.randn(t, h, generator=g, device="cuda").to(torch.bfloat16)
    x_fp8, x_scale = torch.ops.trtllm.fp8_quantize_1x128(x)
    bias = (torch.rand(e, generator=g, device="cuda") - 0.5) * 0.2
    grid = torch.linspace(0.3, 0.7, e, device="cuda")
    order = torch.rand(t, e, generator=g, device="cuda").argsort(dim=1)
    logits = torch.logit(grid[order] - bias).contiguous()
    w31, s31 = fp8(e, 2 * i, h), scales(e, 2 * i // 128, h // 128, fan_in=h)
    w2, s2 = fp8(e, h, i), scales(e, h // 128, i // 128, fan_in=i)
    addend = (torch.randn(t, h, generator=g, device="cuda") * 0.1).to(torch.bfloat16)
    return [
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
    ]


def _check_routing(args: list) -> None:
    """The selected experts in slot order exactly, and their weights within ``_WEIGHT_RTOL``."""
    logits, bias, scaling = args[2], args[3], args[4]
    ids, wts = route(logits, bias, scaling)
    ref_ids, ref_wts = route_reference(logits, bias, scaling)
    assert torch.equal(ids, ref_ids), "a row routes to other experts or slots"
    torch.testing.assert_close(wts, ref_wts, rtol=_WEIGHT_RTOL, atol=0.0, equal_nan=True)


def _check(args: list) -> None:
    """The routing, then the routed experts alone (the addend zeroed: the shared expert's
    output is about 13 times the routed one and would hide a lost pair), then the sum with the
    addend."""
    _check_routing(args)
    routed = list(args)
    routed[10] = torch.zeros_like(args[10])
    _compare(op(*routed), op.reference(*routed))
    _compare(op(*args), op.reference(*args))


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    _check(_build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31))


def test_clamp_is_applied() -> None:
    """A tiny SwiGLU limit makes the clamped and unclamped products disagree; the op must match
    the clamped reference."""
    args = _build(dict(t=2048, **_GLM), seed=7)
    args[5] = 0.05
    _check(args)


def test_skewed_routing() -> None:
    """Most rows on a few experts: segments of many tiles next to empty experts."""
    args = _build(dict(t=2048, **_GLM), seed=11)
    args[2][:, :4] += 100.0
    _check(args)


def test_workspace_serves_a_smaller_step() -> None:
    """A step after a larger one runs on dense views of the larger buffers."""
    workspace = PrefillMoeWorkspace()
    for t, seed in ((2048, 21), (37, 22)):
        args = _build(dict(t=t, **_GLM), seed=seed)
        args[11] = workspace
        with validating(op):
            out = op(*args)
        _compare(out, op.reference(*args))


def test_one_row_scale_from_the_gather() -> None:
    """A one-token prefill: the target all-gathers the scales row-major (a fresh ``[1, K / 128]``
    tensor) and transposes them back, which leaves the size-1 column's stride at K / 128."""
    args = _build(dict(t=1, **_GLM), seed=31)
    gathered = args[1].t().clone(memory_format=torch.contiguous_format)
    args[1] = gathered.t().contiguous()
    assert args[1].shape[1] == 1 and args[1].stride(1) != 1
    with validating(op):
        out = op(*args)
    _compare(out, op.reference(*args))


def test_nan_logits_route_in_range() -> None:
    """A row of NaN router logits must not index past the experts (a device-side assert on every
    rank): it routes to experts 0 to 7 with NaN weights, so its output is NaN. A row with some
    NaN logits routes among its numbers. Every other row is unaffected."""
    args = _build(dict(t=37, **_GLM), seed=23)
    logits = args[2]
    logits[5] = float("nan")
    logits[6, ::2] = float("nan")
    ids, wts = route(logits, args[3], args[4])
    assert torch.equal(ids[5], torch.arange(8, dtype=torch.int32, device="cuda"))
    assert bool(torch.isnan(wts[5]).all())
    assert bool((ids[6] % 2 == 1).all()) and bool(torch.isfinite(wts[6]).all())
    _check_routing(args)
    out, ref = op(*args), op.reference(*args)
    torch.cuda.synchronize()
    assert bool(torch.isnan(out[5]).all()) and bool(torch.isnan(ref[5]).all())
    keep = torch.ones(out.shape[0], dtype=torch.bool, device="cuda")
    keep[5] = False
    _compare(out[keep], ref[keep])
    # The NaN row left nothing behind that breaks the next call.
    clean = _build(dict(t=37, **_GLM), seed=24)
    _compare(op(*clean), op.reference(*clean))


def test_guard_refuses_mismatched_expert_counts() -> None:
    """Expert ids come from the logits' width: a bias, weight or scale of fewer experts is
    indexed past its end."""
    args = _build(dict(t=37, **_GLM), seed=25)
    for index in (3, 6, 7, 8, 9):
        bad = list(args)
        bad[index] = args[index][:-1]
        with pytest.raises(AssertionError):
            with validating(op):
                op(*bad)
