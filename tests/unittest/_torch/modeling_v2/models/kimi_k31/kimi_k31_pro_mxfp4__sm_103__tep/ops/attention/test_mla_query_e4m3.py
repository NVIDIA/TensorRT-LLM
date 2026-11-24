# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 k31_mla_query_e4m3 entry (absorbed query -> per-tensor E4M3 rows), compared
byte for byte."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.attention.mla_query_e4m3 import (  # noqa: E501
    k31_mla_query_e4m3 as op,
)

__extra_import_path__ = ["../../../../..", "../../../ops/attention"]
import _k31_topk_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    n, heads = spec["tokens"], spec["heads"]
    spread = 2000.0 if spec.get("data") == "wide" else 30.0
    fused = (torch.randn(n, heads, 576, generator=gen) * spread).bfloat16().cuda()
    if spec["fused"]:
        q_latent, q_pe = fused[..., :512], fused[..., 512:]
    else:
        q_latent, q_pe = fused[..., :512].contiguous(), fused[..., 512:].contiguous()
    scale = torch.tensor([0.875], dtype=torch.float32, device="cuda")
    out = torch.empty(n, heads, 576, dtype=torch.float8_e4m3fn, device="cuda")
    return dict(q_latent=q_latent, q_pe=q_pe, scale=scale, out=out)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    args = _build(cell.spec, inputs.seed_of(cell))
    ref_out = torch.empty_like(args["out"])
    op(**args)
    op.reference(**dict(args, out=ref_out))
    torch.cuda.synchronize()
    assert torch.equal(args["out"].view(torch.uint8), ref_out.view(torch.uint8))


def test_saturation_keeps_the_sign():
    """Independent of the reference: values past 448 after scaling become +-448, never NaN."""
    args = _build(dict(tokens=4, heads=96, fused=True, data="wide"), seed=2)
    op(**args)
    vals = args["out"].float()
    assert not bool(vals.isnan().any())
    big = (args["q_latent"].float() * 0.875).abs() > 464
    assert bool((vals[..., :512].abs()[big] == 448).all())


def test_guard_refuses_mismatched_rows():
    args = _build(op.CELLS[1].spec, seed=3)
    with pytest.raises(AssertionError, match="rows and heads"):
        with validating(op):
            op(**dict(args, q_pe=args["q_pe"][:-1]))
