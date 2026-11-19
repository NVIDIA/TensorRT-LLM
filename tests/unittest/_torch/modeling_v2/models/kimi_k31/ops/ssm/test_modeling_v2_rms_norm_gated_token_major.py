# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 rms_norm_gated_token_major entry.

Each cell builds the KDA layer's prefill output gate: x the contiguous [T * H, 128] core of the prefill, z the
[T, H, 128] view of the fused [q | k | v | g] projection's gate columns (or a dense gate), the bf16 o_norm weight,
sigmoid gate. Output within one bf16 rounding of an fp32 evaluation.
"""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rms_norm_gated_token_major import (
    rms_norm_gated_token_major as op,
)

from ._kda_ops_util import generator, requires_sm103

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm103

#: A boundary and a typical cell of ``op.CELLS``.
_CELLS = sample_cells(op.CELLS, size_key="tokens")[:2]


def _inputs(spec, gen):
    tokens, heads, width = spec["tokens"], spec["heads"], spec["width"]
    dim = heads * width
    x = torch.randn(tokens * heads, width, generator=gen, device="cuda").to(torch.bfloat16)
    columns = spec["gate_columns_of"]
    # The gate is the last section of a [T, columns * D] projection output.
    projection = torch.randn(tokens, columns * dim, generator=gen, device="cuda").to(torch.bfloat16)
    z = projection[:, (columns - 1) * dim :].reshape(tokens, heads, width)
    weight = (1 + 0.1 * torch.randn(width, generator=gen, device="cuda")).to(torch.bfloat16)
    return x, z, weight, spec["eps"]


@pytest.mark.parametrize("cell", _CELLS, ids=[c.why[:60] for c in _CELLS])
@torch.inference_mode()
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    x, z, weight, eps = _inputs(cell.spec, generator(cell))
    out = op(x, z, weight, eps, None, "sigmoid")
    ref = op.reference(x, z, weight, eps, None, "sigmoid")
    # fp32 statistics in both; one rounding to bf16 at the end (plus the kernel's rsqrt): one bf16 ulp
    # of the output's scale, relative and absolute.
    torch.testing.assert_close(out, ref, rtol=2**-7, atol=2**-7)
