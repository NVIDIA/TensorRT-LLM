# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the cep FP8 all-to-all rows (``rubin_cp_fp8_pack`` / ``rubin_cp_fp8_unpack``).

The pack and unpack are bit-exact against their references; packing the three q / k / v sections of
a cep8 rank's rows for 8 destinations and unpacking one destination's block recovers that
destination's heads within E4M3's rounding (2^-4 of each row group's largest magnitude).
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_cp_fp8_rows import (
    rubin_cp_fp8_pack,
    rubin_cp_fp8_unpack,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="E4M3 Triton conversions",
)


@torch.inference_mode()
def test_pack_unpack_match_their_references() -> None:
    gen = torch.Generator(device="cuda").manual_seed(0)
    rows, groups, gd = 5, 6, 3
    x = (torch.randn(rows, groups * 128, generator=gen, device="cuda") * 3).to(torch.bfloat16)
    row_stride = 2 * (gd * 128 + gd * 4)
    dest_stride = rows * row_stride
    code_off, scale_off = 16, 16 + gd * 128 + 8
    buf = torch.zeros(2 * dest_stride, dtype=torch.uint8, device="cuda")
    ref = buf.clone()
    rubin_cp_fp8_pack(x, buf, dest_stride, row_stride, gd, code_off, scale_off)
    rubin_cp_fp8_pack.reference(x, ref, dest_stride, row_stride, gd, code_off, scale_off)
    torch.cuda.synchronize()
    assert torch.equal(buf, ref)
    out = torch.zeros(rows, 64 + groups * 128, dtype=torch.bfloat16, device="cuda")
    ref_out = out.clone()
    args = (rows, 2, dest_stride, row_stride, gd, code_off, scale_off, 64)
    rubin_cp_fp8_unpack(buf, out, *args)
    rubin_cp_fp8_unpack.reference(buf, ref_out, *args)
    torch.cuda.synchronize()
    assert torch.equal(out, ref_out)
    err = (out[:, 64:].float() - x.float()).abs().view(rows, groups, 128).amax(-1)
    assert (err <= x.float().abs().view(rows, groups, 128).amax(-1) / 16).all()


@torch.inference_mode()
def test_cep8_sections_reach_their_destinations() -> None:
    """Each section of a rank's ``[q | k | v]`` rows goes to the 8 destinations by heads; a
    destination's block unpacks to its heads' ``[q | k | v]``."""
    gen = torch.Generator(device="cuda").manual_seed(1)
    world, rows, heads = 8, 33, 96
    p, pc = heads * 128, heads * 128 // world
    gd = pc // 128
    y = torch.randn(rows, 3 * p + 40, generator=gen, device="cuda").to(torch.bfloat16)
    wb = -(-(3 * pc + 3 * gd * 4 + 64) // 16) * 16
    send = torch.empty(world, rows, wb, dtype=torch.uint8, device="cuda")
    for s in range(3):
        rubin_cp_fp8_pack(
            y[:, s * p : (s + 1) * p], send, rows * wb, wb, gd, s * pc, 3 * pc + s * gd * 4
        )
    for j in (0, 5, 7):
        got = torch.empty(rows, 3 * pc, dtype=torch.bfloat16, device="cuda")
        for s in range(3):
            rubin_cp_fp8_unpack(
                send[j], got, rows, 1, 0, wb, gd, s * pc, 3 * pc + s * gd * 4, s * pc
            )
        want = y[:, : 3 * p].view(rows, 3, world, pc)[:, :, j].reshape(rows, 3 * pc)
        torch.cuda.synchronize()
        assert torch.allclose(got.float(), want.float(), rtol=2**-3, atol=1e-2)


def _packed(rows: int, groups: int, gd: int, gen: torch.Generator) -> tuple:
    """BF16 rows of ``groups`` 128-column groups, packed for ``groups // gd`` destinations (codes
    then scales per row)."""
    x = (torch.randn(rows, groups * 128, generator=gen, device="cuda") * 3).to(torch.bfloat16)
    row_stride = gd * 128 + gd * 4
    dest_stride = rows * row_stride
    buf = torch.zeros(groups // gd * dest_stride, dtype=torch.uint8, device="cuda")
    return x, buf, dest_stride, row_stride


@pytest.mark.parametrize("cell", rubin_cp_fp8_pack.CELLS, ids=lambda c: c.why)
@torch.inference_mode()
def test_pack_cells_match_the_reference(cell) -> None:
    rows, groups, gd, dests = (cell.spec[k] for k in ("rows", "groups", "gd", "dests"))
    assert groups // gd == dests
    gen = torch.Generator(device="cuda").manual_seed(rows + groups)
    x, buf, dest_stride, row_stride = _packed(rows, groups, gd, gen)
    ref = buf.clone()
    rubin_cp_fp8_pack(x, buf, dest_stride, row_stride, gd, 0, gd * 128)
    rubin_cp_fp8_pack.reference(x, ref, dest_stride, row_stride, gd, 0, gd * 128)
    torch.cuda.synchronize()
    assert torch.equal(buf, ref)


@pytest.mark.parametrize("cell", rubin_cp_fp8_unpack.CELLS, ids=lambda c: c.why)
@torch.inference_mode()
def test_unpack_cells_match_the_reference(cell) -> None:
    rows, gd, num_src = (cell.spec[k] for k in ("rows", "gd", "num_src"))
    gen = torch.Generator(device="cuda").manual_seed(rows + num_src)
    x, buf, dest_stride, row_stride = _packed(rows, num_src * gd, gd, gen)
    rubin_cp_fp8_pack(x, buf, dest_stride, row_stride, gd, 0, gd * 128)
    out = torch.zeros(rows, num_src * gd * 128, dtype=torch.bfloat16, device="cuda")
    ref_out = out.clone()
    args = (rows, num_src, dest_stride, row_stride, gd, 0, gd * 128, 0)
    rubin_cp_fp8_unpack(buf, out, *args)
    rubin_cp_fp8_unpack.reference(buf, ref_out, *args)
    torch.cuda.synchronize()
    assert torch.equal(out, ref_out)
