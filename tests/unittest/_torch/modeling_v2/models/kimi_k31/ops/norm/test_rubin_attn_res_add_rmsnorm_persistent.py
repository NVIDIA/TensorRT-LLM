# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 rubin_attn_res_add_rmsnorm_persistent entry.

A cell is (T, N, add): T tokens, N = K + 1 candidates, with or without the residual add. Every
cell checks shapes and dtypes, bit-identical repeated calls, untouched inputs, the updated prefix
bit for bit (a torch bf16 add; without an add, the input itself), and the output in bf16 steps:
against the entry's reference (one step) and against the per-token C++ op of the same contract
(two steps: that op rounds twice more).
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_add_rmsnorm_persistent import (  # noqa: E501
    rubin_attn_res_add_rmsnorm_persistent as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="certified on sm_107 (VR200)",
)

H = 7168
EPS = 1e-5
# A step is one bf16 ulp (8 significant bits) at the larger of the two values, taken at no less
# than 1/32 of the row RMS: where the mixture cancels, the fp32 summation order alone moves an
# output by more than its own ulp.
FLOOR = 1 / 32
# Against the reference both sides round once: one step at most, and few elements move (fp32
# differences of about 1e-6 flip about 1e-6 / 2^-8 of the roundings). One extra rounding point
# moves about a third of them.
MAX_CHANGED = 0.01
# Against the per-token op, which also rounds the mixture and the normalized value.
PER_TOKEN_STEPS = 2


def _inputs(tokens: int, candidates: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rn(*shape, sc=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * sc).to(torch.bfloat16)

    return (
        rn(tokens, 1, H, sc=0.05),
        rn(tokens, 1, H, sc=0.05),
        rn(candidates - 1, tokens, 1, H, sc=0.05),
        rn(H, sc=0.02),
        (1 + rn(H, sc=0.02).float()).to(torch.bfloat16),
        (1 + rn(H, sc=0.02).float()).to(torch.bfloat16),
    )


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _steps(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """|actual - expected| per element, in bf16 steps (see FLOOR)."""
    a, e = actual.float(), expected.float()
    floor = e.square().mean(dim=-1, keepdim=True).sqrt() * FLOOR
    scale = torch.maximum(torch.maximum(a.abs(), e.abs()), floor).clamp_min(2.0**-126)
    return (a - e).abs() / torch.exp2(torch.floor(torch.log2(scale)) - 7)


def _assert_steps(
    actual: torch.Tensor,
    expected: torch.Tensor,
    what: str,
    max_steps: int = 1,
    max_changed: float = MAX_CHANGED,
) -> None:
    worst = _steps(actual, expected).max().item()
    changed = (actual.float() != expected.float()).float().mean().item()
    assert worst <= max_steps and changed <= max_changed, (
        f"{what}: {worst:g} bf16 steps (band {max_steps}), {changed:.3%} of the elements changed"
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:56] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(op)
    spec = cell.spec
    assert spec["hidden"] == H
    layer, add, block, res_w, rms_w, out_w = _inputs(
        spec["tokens"], spec["candidates"], zlib.crc32(cell.why.encode()) % 2**31
    )
    add = add if spec["add"] else None
    args = (layer, add, block, res_w, rms_w, out_w, spec["eps"], spec["eps"])
    before = [t.clone() for t in (layer, block, res_w, rms_w, out_w)]
    updated, out = op(*args)
    updated2, out2 = op(*args)
    ref_updated, ref_out = op.reference(*args)
    assert out.shape == layer.shape and out.dtype == torch.bfloat16
    assert torch.equal(_bits(out), _bits(out2)), "two identical calls disagree"
    if add is None:
        assert updated.data_ptr() == layer.data_ptr(), "without an add the input comes back"
    else:
        assert torch.equal(_bits(updated), _bits(updated2))
        assert torch.equal(_bits(updated), _bits(ref_updated)), "the add is not a bf16 add"
    _assert_steps(out, ref_out, "output")
    if add is not None:
        _, cpp = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(*args)
    else:
        cpp = torch.ops.trtllm.attn_res_rmsnorm_fwd(layer, block, res_w, rms_w, out_w, EPS, EPS)
    _assert_steps(out, cpp, "output vs the per-token op", PER_TOKEN_STEPS, max_changed=1.0)
    for old, new in zip(before, (layer, block, res_w, rms_w, out_w)):
        assert torch.equal(_bits(old), _bits(new)), "an input was mutated"


def test_band_refuses_a_dropped_candidate() -> None:
    layer, add, block, res_w, rms_w, out_w = _inputs(96, 6, seed=5)
    _, out = op(layer, add, block, res_w, rms_w, out_w, EPS, EPS)
    _, wrong = op.reference(layer, add, block[1:], res_w, rms_w, out_w, EPS, EPS)
    with pytest.raises(AssertionError):
        _assert_steps(out, wrong, "dropped candidate")


def test_band_refuses_the_per_token_rounding() -> None:
    """The reference band tells one rounding from the per-token op's three."""
    layer, add, block, res_w, rms_w, out_w = _inputs(96, 6, seed=5)
    args = (layer, add, block, res_w, rms_w, out_w, EPS, EPS)
    _, out = op(*args)
    _, cpp = torch.ops.trtllm.attn_res_add_rmsnorm_fwd(*args)
    with pytest.raises(AssertionError):
        _assert_steps(out, cpp, "the per-token rounding")
