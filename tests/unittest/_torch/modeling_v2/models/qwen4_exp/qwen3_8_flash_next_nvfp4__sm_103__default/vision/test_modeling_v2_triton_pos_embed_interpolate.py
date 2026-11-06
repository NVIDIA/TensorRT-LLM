# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the default target's triton_pos_embed_interpolate entry (its vision
tower).

`op.reference` is the explicit four-corner sum in fp64, returned unrounded as
fp32, so the band measures the op's own bf16 arithmetic. `_resize64` runs
torch's `F.interpolate` in fp64 and shares no arithmetic with either: each cell
holds the reference to it (the structure: which axis is which, corner
alignment, merge row order), and its wrong variants (`align_corners=False`,
raster rows, swapped table axes) must fail the band.
"""

import zlib

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.vision.triton_pos_embed_interpolate import (  # noqa: E501
    _bilinear_pos_embed_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__default.ops.vision.triton_pos_embed_interpolate import (  # noqa: E501
    triton_pos_embed_interpolate as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import ULP, assert_within_ulp, certifying, validating  # noqa: E402,F401

assert torch.cuda.is_available(), "triton_pos_embed_interpolate requires a CUDA device"

NUM_GRID = 48
HIDDEN = 1152
MERGE = 2


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 bf16 ulp per element against the row scale, 1 ulp relative RMS.

    The op rounds its four corner weights to bf16 and multiply-adds in bf16 (the
    HF reference's arithmetic); `ref` is the exact value. The worst element
    reaches about 3.5 ulp and the RMS 0.75, on `_table` and on the checkpoint's
    own table alike. The nearest wrong variant, `align_corners=False`, lands an
    order of magnitude outside the band, raster order and swapped axes two.
    """
    assert out.dtype == torch.bfloat16, out.dtype
    assert_within_ulp(out.float(), ref, element_ulp=4.0, rms_ulp=1.0, ulp=ULP[torch.bfloat16])


def _table(seed: int, num_grid: int = NUM_GRID, hidden: int = HIDDEN) -> torch.Tensor:
    """A smooth `[num_grid**2, hidden]` bf16 table, the way a learned 2-D
    position table is: a 12x12 normal field upsampled bicubically, plus 0.1
    noise. Neighbouring rows have cosine 0.911; the checkpoint table's have 0.907.

    Contiguous on purpose: the upsampled field's permuted view reshapes to a
    strided table, which the kernel reads wrongly (`is_valid` refuses it).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    coarse = torch.randn(1, hidden, 12, 12, generator=g, device="cuda")
    up = F.interpolate(coarse, size=(num_grid, num_grid), mode="bicubic", align_corners=True)
    up = up[0].permute(1, 2, 0) + 0.1 * torch.randn(
        num_grid, num_grid, hidden, generator=g, device="cuda"
    )
    return up.reshape(num_grid * num_grid, hidden).to(torch.bfloat16).contiguous()


def _resize64(table, t, h, w, m_size, *, align=True, raster=False, swap=False):
    """torch's bilinear resize in fp64, in merge order unless `raster`."""
    num_grid = int(round(table.shape[0] ** 0.5))
    grid = table.double().view(num_grid, num_grid, -1)
    if swap:
        grid = grid.transpose(0, 1)
    src = grid.permute(2, 0, 1).unsqueeze(0).contiguous()
    out = F.interpolate(src, size=(h, w), mode="bilinear", align_corners=align)[0].permute(1, 2, 0)
    if raster:
        rows = out.reshape(h * w, -1)
    else:
        rows = (
            out.reshape(h // m_size, m_size, w // m_size, m_size, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(h * w, -1)
        )
    return rows.repeat(t, 1) if t > 1 else rows


def _args(spec, table):
    return (
        table,
        spec["t"],
        spec["h"],
        spec["w"],
        spec["num_grid_per_side"],
        spec["m_size"],
        spec["dtype"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    t, h, w, m_size = spec["t"], spec["h"], spec["w"], spec["m_size"]
    table = _table(zlib.crc32(cell.why.encode()) % 2**31, spec["num_grid_per_side"], spec["hidden"])
    before = table.clone()

    out = op(*_args(spec, table))
    assert out.shape == (t * h * w, spec["hidden"]) and out.dtype == spec["dtype"]
    assert torch.equal(table, before) and out.data_ptr() != table.data_ptr()

    ref = op.reference(*_args(spec, table))
    _compare(out, ref)

    # The reference's structure, from an implementation that shares nothing with it.
    torch.testing.assert_close(ref, _resize64(table, t, h, w, m_size).float())

    # The band refuses each wrong variant of the function.
    for wrong in (dict(align=False), dict(raster=True), dict(swap=True)):
        with pytest.raises(AssertionError):
            _compare(out, _resize64(table, t, h, w, m_size, **wrong).float())


def test_one_compile_serves_every_grid():
    """The grid sides are not specialized: sides that are and are not multiples of 16 run
    the one compiled kernel, each within the band."""
    table = _table(41)
    variants = []
    for t, h, w in ((1, 2, 2), (1, 16, 16), (1, 4, 34), (2, 62, 90), (1, 32, 6)):
        args = (table, t, h, w, NUM_GRID, MERGE, torch.bfloat16)
        _compare(op(*args), op.reference(*args))
        variants.append(
            len(_bilinear_pos_embed_kernel.device_caches[torch.cuda.current_device()][0])
        )
    assert variants == [variants[0]] * len(variants), variants
