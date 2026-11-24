# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the pos_embed_interpolate entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.ops.vision.pos_embed_interpolate import (  # noqa: E501
    pos_embed_interpolate as op,
)

__extra_import_path__ = ["~/tests/unittest/_torch/modeling_v2"]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "pos_embed_interpolate requires a CUDA device"

# fp32 blend vs fp64 blend, each rounded once: a boundary flip reads up to 2 ulp of the row scale.
_ELEMENT_ULP = 3.0
_RMS_ULP = 1.0


def _seed(text: str) -> int:
    return zlib.crc32(text.encode())


def _table(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = spec["position_edge"] ** 2 + 1
    return torch.randn(rows, spec["hidden"], generator=g, device="cuda").to(spec["dtype"])


def _hf_path(table: torch.Tensor, t: int, h: int, w: int, edge: int, dtype) -> torch.Tensor:
    """`HYV4VLVisionPatchEmbed.interpolate_pos_encoding`, the HF tower's math verbatim."""
    pos = table[1:].reshape(1, edge, edge, -1).permute(0, 3, 1, 2).float()
    pos = torch.nn.functional.interpolate(
        pos,
        scale_factor=((h + 0.1) / edge, (w + 0.1) / edge),
        mode="bilinear",
        align_corners=False,
    )
    return pos.permute(0, 2, 3, 1).reshape(h * w, -1).to(dtype).repeat(t, 1)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    s = cell.spec
    table = _table(s, seed=_seed(cell.why))
    result = op(table, s["t"], s["h"], s["w"], s["position_edge"], s["dtype"])
    ref = op.reference(table, s["t"], s["h"], s["w"], s["position_edge"], s["dtype"])
    assert_within_ulp(result, ref, _ELEMENT_ULP, _RMS_ULP)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_matches_the_hf_reference(cell) -> None:
    """The HF path through `F.interpolate`, whose identity at the table's own size the reference mirrors by hand."""
    s = cell.spec
    table = _table(s, seed=_seed(cell.why) + 1)
    result = op(table, s["t"], s["h"], s["w"], s["position_edge"], s["dtype"])
    hf = _hf_path(table, s["t"], s["h"], s["w"], s["position_edge"], s["dtype"])
    assert_within_ulp(result, hf, _ELEMENT_ULP, _RMS_ULP)
