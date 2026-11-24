# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_rms_norm_gated_token_major entry (sm_107).

Each cell builds the context KDA output gate call: x the [T * H, 128] bf16 core, z the gate columns
of the fused [q | k | v | g] projection (or of the specialized targets' stacked [q | k | v | g | f_a | b]
projection) read token-major. The output within one bf16 rounding of
the entry's fp32 reference; the inputs untouched. A prefetch cell's output bit for bit the same call's
without prefetch, and so are rows of zeros, tiny, huge, infinite and NaN values at prefill size.
"""

import zlib

import pytest
import torch
from _rubin_kda_prefill_inputs import stacked_columns, stacked_projection

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_rms_norm_gated_token_major import (
    rubin_rms_norm_gated_token_major as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin entry: certified on sm_107",
)


def norm_args(spec: dict, seed: int, device="cuda") -> dict:
    """x, the z view (`gate_columns_of` = 4: the gate columns of a [T, 4 * H * N] projection; 1: a
    dense [T, H, N] gate; `stacked`: the g columns of `stacked_projection`), the weight, eps, the
    optional FP8 scale and the KDA gate."""
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens, heads, width = spec["tokens"], spec["heads"], spec["width"]
    if spec.get("stacked"):
        g0 = stacked_columns(heads)["g"]
        proj = stacked_projection(tokens, heads, gen, device)
        z = proj[:, g0 : g0 + heads * width].view(tokens, heads, width)
    else:
        sections = spec["gate_columns_of"]
        proj = torch.randn(tokens, sections * heads * width, generator=gen, device=device)
        z = proj.to(torch.bfloat16)[:, (sections - 1) * heads * width :].view(tokens, heads, width)
    scale = spec.get("fp8_scale")
    return dict(
        x=(torch.randn(tokens * heads, width, generator=gen, device=device) * 3).to(torch.bfloat16),
        z=z,
        weight=(1 + 0.1 * torch.randn(width, generator=gen, device=device)).to(torch.bfloat16),
        eps=spec["eps"],
        fp8_scale=None if scale is None else torch.tensor([scale], device=device),
        gate_activation="sigmoid",
        prefetch=spec.get("prefetch", False),
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:60] for c in op.CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = norm_args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    before = {k: v.clone() for k, v in args.items() if isinstance(v, torch.Tensor)}
    out = op(**args)
    ref = op.reference(**args)
    torch.cuda.synchronize()
    assert out.dtype == ref.dtype and out.shape == ref.shape and out.is_contiguous()
    if args["fp8_scale"] is None:
        # Kernel and reference differ only in the order of the row's sum of squares: within one
        # bf16 rounding of each element.
        torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-7, atol=1e-3)
    else:
        # One e4m3 step at most where the bf16 intermediates round apart.
        torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-3, atol=2.0**-6)
    for name, t in before.items():
        assert torch.equal(args[name], t), f"{name} written"


@pytest.mark.parametrize(
    "cell",
    [c for c in op.CELLS if c.spec.get("prefetch")],
    ids=[c.why[:60] for c in op.CELLS if c.spec.get("prefetch")],
)
@torch.inference_mode()
def test_prefetch_is_bitwise_the_plain_kernel(cell) -> None:
    args = norm_args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = op(**args)
    plain = op(**{**args, "prefetch": False})
    torch.cuda.synchronize()
    assert out.dtype == plain.dtype and out.shape == plain.shape
    assert torch.equal(out.view(torch.uint8), plain.view(torch.uint8))


def _edge_args(tokens: int, heads: int, seed: int) -> dict:
    """Rows whose gated values are zero, tiny, huge, infinite or NaN; the rest ordinary. Row
    ``t * heads + h``."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens * heads, 128, generator=gen, device="cuda") * 3
    z = torch.randn(tokens, heads, 128, generator=gen, device="cuda") * 2
    x[0] = 0  # an all-zero row: rstd = eps ** -0.5, every value 0
    z[1] = -1.0e4  # a closed gate: every value 0
    for h in range(heads):  # gates near -88: values near the bottom of the BF16 range
        z[2, h] = -88.0 + 0.25 * (h % 9)
    x[4 * heads] = 3.0e38  # an overflowing sum of squares: rstd 0
    x[4 * heads + 1, :32] = 1.0e-38  # BF16 subnormal inputs
    x[5 * heads, 7] = float("inf")  # an infinite input: a NaN row
    x[5 * heads + 1, 9] = float("nan")  # a NaN input: a NaN row
    z[6, 0, :32] = float("nan")
    z[7, 0, 5] = float("inf")
    z[7, 1, 6] = -float("inf")
    return dict(
        x=x.to(torch.bfloat16),
        z=z.to(torch.bfloat16),
        weight=(1 + 0.1 * torch.randn(128, generator=gen, device="cuda")).to(torch.bfloat16),
        eps=1e-5,
        gate_activation="sigmoid",
    )


@pytest.mark.parametrize("tokens, heads", [(2953, 96), (8191, 12)])
@torch.inference_mode()
def test_prefetch_edge_rows_bitwise_the_plain_kernel(tokens, heads) -> None:
    args = _edge_args(tokens, heads, seed=tokens + heads)
    out = op(**args, prefetch=True)
    plain = op(**args)
    torch.cuda.synchronize()
    assert bool(plain.isnan().any()), "no NaN row"
    assert bool((plain.view(tokens, -1) == 0).all(dim=1).any()), "no all-zero token row"
    assert torch.equal(out.view(torch.uint8), plain.view(torch.uint8))


def test_guard_refuses_a_per_column_fp8_scale() -> None:
    args = norm_args(
        dict(tokens=4, heads=12, width=128, eps=1e-5, gate_columns_of=4, fp8_scale=0.05), seed=0
    )
    args["fp8_scale"] = torch.full((128,), 0.05, device="cuda")
    with pytest.raises(AssertionError, match="exactly one element"):
        with validating(op):
            op(**args)
