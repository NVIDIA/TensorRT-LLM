# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attn_res_add_rmsnorm_persistent_fwd entry (engine persistent attn_res op)."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm._attn_res_common import (  # noqa: E501
    HIDDEN,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_add_rmsnorm_persistent_fwd import (  # noqa: E501
    attn_res_add_rmsnorm_persistent_fwd as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "the persistent attn_res op requires a CUDA device"
if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
    pytest.skip("the engine attn_res ops run on the sm_100 family", allow_module_level=True)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)
RMS_EPS, OUT_EPS = 1e-6, 1e-5


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    m, n = spec["rows"], spec["candidates"]

    def r(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)

    prefix, block = r(m, HIDDEN, scale=4.0), r(n - 1, m, HIDDEN, scale=4.0)
    addend = r(m, HIDDEN) if spec["add"] else None
    weights = (r(HIDDEN, scale=0.05), (1 + r(HIDDEN, scale=0.1).float()).bfloat16(), r(HIDDEN))
    return prefix, addend, block, weights


def _close(got: torch.Tensor, ref: torch.Tensor, out_w: torch.Tensor) -> None:
    """One BF16 step at the row's scale per rounding (mixture, normalized row, weight product)."""
    g, w = got.float(), ref.float()
    weight = out_w.float().abs()

    def ulp(x):
        return torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(2.0**-126))) - 7)

    normed = (w.abs() / weight.clamp_min(2.0**-20)).clamp_min(1.0)
    tol = 2 * ulp(normed) * weight + ulp(torch.maximum(g.abs(), w.abs()))
    diff = (g - w).abs()
    assert torch.all(diff <= tol * 1.0001), f"max diff {diff.max().item()}"


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:40] + str(c.spec) for c in CELLS])
def test_certified_cells(cell) -> None:
    prefix, addend, block, weights = _build(
        cell.spec, seed=zlib.crc32(repr(cell.spec).encode()) % 2**31
    )
    args = (prefix, addend, block, *weights, RMS_EPS, OUT_EPS)
    with validating(op):
        updated, out = op(*args)
    ref_updated, ref_out = op.reference(*args)
    assert torch.equal(updated.view(torch.int16), ref_updated.view(torch.int16))
    _close(out, ref_out, weights[2])


def test_guard_refuses_ten_candidates() -> None:
    prefix, addend, block, weights = _build(dict(rows=4, candidates=9, add=True), seed=1)
    block = torch.cat([block, block[:1]], dim=0)
    with pytest.raises(AssertionError, match="2..9 candidates"):
        with validating(op):
            op(prefix, addend, block, *weights, RMS_EPS, OUT_EPS)
