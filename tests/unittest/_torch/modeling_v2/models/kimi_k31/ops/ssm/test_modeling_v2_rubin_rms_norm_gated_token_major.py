# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_rms_norm_gated_token_major entry (sm_107).

Each cell builds the context KDA output gate call: x the [T * H, 128] bf16 core, z the gate columns
of the fused [q | k | v | g] projection (or of the specialized targets' stacked [q | k | v | g | f_a | b]
projection) read token-major. The output within one bf16 rounding of
the entry's fp32 reference.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rubin_rms_norm_gated_token_major import (
    rubin_rms_norm_gated_token_major as op,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _kda_ops_util import (  # noqa: E402 — needs the path declared above
    stacked_columns,
    stacked_projection,
)
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm(107)

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="tokens")[:2]


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
    )


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:60] for c in _CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = norm_args(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = op(**args)
    ref = op.reference(**args)
    torch.cuda.synchronize()
    if args["fp8_scale"] is None:
        # Kernel and reference differ only in the order of the row's sum of squares: within one
        # bf16 rounding of each element.
        torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-7, atol=1e-3)
    else:
        # One e4m3 step at most where the bf16 intermediates round apart.
        torch.testing.assert_close(out.float(), ref.float(), rtol=2.0**-3, atol=2.0**-6)
