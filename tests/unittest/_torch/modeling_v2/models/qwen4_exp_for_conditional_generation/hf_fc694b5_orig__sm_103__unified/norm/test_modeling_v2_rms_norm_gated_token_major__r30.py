# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rms_norm_gated_token_major entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.norm.rms_norm_gated_token_major__r30 import (  # noqa: E501
    rms_norm_gated_token_major as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "rms_norm_gated_token_major requires a CUDA device"


#: The width of the concatenated GDN input projection `[q k v | z | a | b]` a cell names no
#: `z_row_stride` for: all 48 heads, 10240 | 6144 | 48 | 48. A tp4/tep4 rank's is 4120:
#: 2560 | 1536 | 12 | 12.
_IN_PROJ_WIDTH = 16480


def _z_offset(row_stride: int, heads: int, n: int) -> int:
    """First column of `z` in a `[q k v | z | a | b]` row: `a` and `b` are one column a head."""
    offset = row_stride - heads * (n + 2)
    assert offset > 0, (row_stride, heads, n)
    return offset


#: The checkpoint's GDN epilogue geometry over a packed `z`, for the guard test.
_PACKED = dict(op.CELLS[0].spec, tokens=4, z="packed")


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _build(spec, seed):
    """`__call__`'s positional arguments for one spec."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    t, heads, n, dtype = spec["tokens"], spec["heads"], spec["width"], spec["dtype"]

    def randn(*shape):
        return torch.randn(*shape, generator=g, device="cuda").to(dtype)

    x = randn(t * heads, n)

    layout = spec.get("z", "packed")
    if layout == "in_proj":
        width = spec.get("z_row_stride", _IN_PROJ_WIDTH)
        offset = _z_offset(width, heads, n)
        z = randn(t, width)[:, offset : offset + heads * n].view(t, heads, n)
    else:
        assert layout == "packed", layout
        z = randn(t, heads, n)

    weight = randn(n)
    return x, z, weight, spec["eps"], None, spec["gate_activation"]


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell.why))
    x, z, weight = args[:3]
    assert z.stride()[1:] == (cell.spec["width"], 1)
    if cell.spec["tokens"] > 1:
        # A one-token view reports the dense stride for its size-1 token dim (and is then
        # contiguous); the kernel never reads past token 0 there.
        row_stride = cell.spec.get("z_row_stride", _IN_PROJ_WIDTH)
        assert z.stride(0) == row_stride and not z.is_contiguous()
    before = [x.clone(), z.clone(), weight.clone()]

    out = op(*args)

    assert out.shape == x.shape and out.is_contiguous() and out.dtype == x.dtype
    for t, b in zip((x, z, weight), before):
        assert torch.equal(t, b), "inputs are read-only"
    torch.testing.assert_close(out, op.reference(*args))


def test_guard_refuses_a_weight_of_the_wrong_length() -> None:
    """Unguarded, a length-192 weight is silently truncated to weight[:128]."""
    x, z, weight, eps, _, act = _build(_PACKED, seed=12)
    long_weight = torch.cat([weight, torch.randn(64, dtype=weight.dtype, device="cuda")])
    assert torch.equal(
        torch.ops.trtllm.rms_norm_gated_token_major(x, z, long_weight, eps, None, act),
        torch.ops.trtllm.rms_norm_gated_token_major(x, z, weight, eps, None, act),
    ), "the hazard this guard covers is gone"

    with pytest.raises(AssertionError, match="weight must be 1-D of length"):
        with validating(op):
            op(x, z, long_weight, eps, None, act)
