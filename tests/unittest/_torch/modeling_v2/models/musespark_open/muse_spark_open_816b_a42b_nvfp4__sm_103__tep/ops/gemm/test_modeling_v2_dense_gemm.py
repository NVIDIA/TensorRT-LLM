# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-GPU test for the tep8 wide dense GEMM cells (``wide_dense_gemms``, ``wide_projection_gathers``).

At tep8 the 9..32-row calls of the QKV projection (``dense_gemms.SM103_TP8_WIDE_KERNELS``) and of
the pooler and unpooler slices (``wide_projection_gathers.TP8_WIDE_KERNELS``) run flashinfer's
split-K with the ``split_k`` of the site's 8-row cell (``dense_gemms.DEFAULT_KERNELS``). They rest
on one property of that kernel: split-K tactics of one ``split_k`` sum the same K slices in the same
order whatever their tile widths, and a row's sum never involves another row, so every row of a
wide call gets the bits it gets in a one-request (8-row) step. This test pins that property on the
served cells, bit for bit, so a kernel change that breaks it fails here rather than silently
changing a decode step's numerics; it also holds every cell to the op's float64 reference.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
    dispatcher as dense_gemms,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.wide_projection_gathers import (  # noqa: E501
    gathers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.gemm import (  # noqa: E501
    dense_gemm as op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="the sm103 target's dense GEMM kernels need an SM 10.3 GPU",
)

#: Per-rank ``(K, N)`` of each tep8 site with wide cells.
SHAPES = {"qkv": op.QKV_SHAPE, "pool": op.POOL_SHAPE, "unpool": op.UNPOOL_SHAPE}
#: The 9..32-row cell of each site, by ``rows - 9``.
WIDE = {
    **dense_gemms.wide_kernels(dense_gemms.SM103_TP8_WIDE_KERNELS),
    **dense_gemms.wide_kernels(gathers.TP8_WIDE_KERNELS),
}
#: The 1..8-row cells of each site at tep8 with the sharded pooler and unpooler.
NARROW = dense_gemms.default_kernels(tp_size=8, sharded=True, capability=(10, 3))
ROWS = (9, 15, 16, 17, 24, 31, 32)


def _operands(site, rows, seed):
    k, n = SHAPES[site]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, k, generator=gen, device="cuda").to(torch.bfloat16)
    weight = (torch.randn(n, k, generator=gen, device="cuda") * k**-0.5).to(torch.bfloat16)
    return x, weight


def test_wide_cells_keep_the_eight_row_split_k():
    assert set(WIDE) == set(SHAPES)
    assert dense_gemms.wide_kernel_table((10, 3), 8, sharded=True, kernels=NARROW) == {
        "qkv": WIDE["qkv"]
    }
    for site, cells in WIDE.items():
        split_k = op.parse_kernel(NARROW[site][op.MAX_ROWS - 1])[1][2]
        k, n = SHAPES[site]
        for rows, kernel in enumerate(cells, start=op.MAX_ROWS + 1):
            family, fields = op.parse_kernel(kernel)
            assert family == op.SPLITK and fields[2] == split_k, (site, rows, kernel)
            assert op.kernel_error(kernel, rows, k, n) is None, (site, rows, kernel)


@pytest.mark.parametrize("enable_pdl", [True, False], ids=["pdl", "no-pdl"])
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("site", sorted(SHAPES))
def test_wide_rows_get_their_eight_row_bits(site, rows, enable_pdl):
    x, weight = _operands(site, rows, seed=rows)
    wide = op.dense_gemm(x, weight, WIDE[site][rows - op.MAX_ROWS - 1], enable_pdl=enable_pdl)
    narrow_cell = NARROW[site][op.MAX_ROWS - 1]
    narrow = torch.cat(
        [
            op.dense_gemm(x[i : i + op.MAX_ROWS], weight, narrow_cell, enable_pdl=enable_pdl)
            for i in range(0, rows, op.MAX_ROWS)
        ]
    )
    torch.cuda.synchronize()
    assert wide.shape == narrow.shape == (rows, SHAPES[site][1])
    mismatched = (wide.view(torch.int16) != narrow.view(torch.int16)).any(dim=1)
    assert not mismatched.any(), f"rows {mismatched.nonzero().flatten().tolist()} differ"


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("site", sorted(SHAPES))
def test_wide_cells_against_the_reference(site, rows):
    x, weight = _operands(site, rows, seed=100 + rows)
    kernel = WIDE[site][rows - op.MAX_ROWS - 1]
    out = op.dense_gemm(x, weight, kernel)
    reference = op.dense_gemm.reference(x, weight, kernel)
    # FP32 accumulation rounded once to BF16 against float64 rounded once: at most one
    # BF16 rounding step apart.
    torch.testing.assert_close(out.float(), reference.float(), rtol=2**-7, atol=2**-14)
