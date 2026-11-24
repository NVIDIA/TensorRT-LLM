# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 fused_cat_fp8 entry (indexer Q / K concatenate + FP8 quantize).

The inputs are built the way the K3.1 indexer builds them: the 128-wide rows of one bf16 tensor
(Q [T, 32, 128], K [T, 128]) split into two 64-wide halves, so pe and nope are strided views.
With UE8M0 scales the output must equal the reference bit for bit; without, each element may
move by one E4M3 step (the kernel's approximate reciprocal).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.fused_cat_fp8 import (  # noqa: E501
    fused_cat_fp8 as op,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103


def _rows(spec, gen) -> torch.Tensor:
    shape = (spec["tokens"], spec["heads"], 128) if spec["heads"] else (spec["tokens"], 128)
    x = torch.randn(*shape, generator=gen)
    if spec["data"] == "zeros_row":
        x.view(-1, 128)[::3] = 0
    elif spec["data"] == "wide":
        x = x * torch.pow(10.0, torch.randint(-6, 7, shape[:-1], generator=gen).float())[..., None]
    return x.to(torch.bfloat16).cuda()


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    x = _rows(spec, gen)
    pe, nope = x.split([spec["pe_dim"], 128 - spec["pe_dim"]], dim=-1)
    return dict(pe=pe, nope=nope, use_ue8m0=spec["use_ue8m0"])


def _e4m3_step(v: torch.Tensor) -> torch.Tensor:
    """The spacing of E4M3 values at |v| (2^-9 in the subnormal range)."""
    a = v.abs().clamp_min(2.0**-6)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - 3)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    out, scale = op(**args)
    ref_out, ref_scale = op.reference(**args)
    torch.cuda.synchronize()
    rows = args["pe"].numel() // args["pe"].shape[-1]
    assert out.shape == (rows, 128) and out.dtype == torch.float8_e4m3fn
    assert scale.shape == (rows, 1) and scale.dtype == torch.float32
    assert torch.equal(scale, ref_scale), "the per-row scale differs"
    if args["use_ue8m0"]:
        assert torch.equal(out.view(torch.uint8), ref_out.view(torch.uint8))
        return
    a, b = out.float(), ref_out.float()
    step = torch.maximum(_e4m3_step(a), _e4m3_step(b))
    assert ((a - b).abs() <= step).all(), "an element moved by more than one E4M3 step"


def test_guard_refuses_rows_that_do_not_collapse():
    x = torch.randn(4, 64, 128, dtype=torch.bfloat16, device="cuda")
    heads = x[:, :32]  # 32 of 64 heads: stride(0) is 64 rows, not 32
    pe, nope = heads.split([64, 64], dim=-1)
    with pytest.raises(AssertionError, match="collapse"):
        with validating(op):
            op(pe=pe, nope=nope, use_ue8m0=True)


def test_rows_that_do_not_collapse_are_read_wrongly():
    """The measurement behind the guard: a head slice is quantized from the wrong rows."""
    x = torch.randn(4, 64, 128, dtype=torch.bfloat16, device="cuda")
    pe, nope = x[:, :32].split([64, 64], dim=-1)
    out, _ = op(pe=pe, nope=nope, use_ue8m0=True)
    ref, _ = op.reference(pe=pe, nope=nope, use_ue8m0=True)
    torch.cuda.synchronize()
    assert not torch.equal(out.view(torch.uint8), ref.view(torch.uint8))
