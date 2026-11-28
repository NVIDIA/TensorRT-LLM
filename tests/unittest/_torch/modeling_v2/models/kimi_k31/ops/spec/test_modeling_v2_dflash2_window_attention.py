# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 window-attention entry (TRTLLM-Gen sparse MLA).

The index lists come from the window-rows entry over random page tables, so every query row
attends its exact window (short contexts floor it at position 0). The kernel rounds the
probabilities and the output to BF16 where the FP32 reference does not, so the gate is a band
in units of the output's BF16 ulp, derived below.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_attention import (  # noqa: E501
    dflash2_window_attention as op,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_attention import (  # noqa: E501
    dflash2_window_attention_buffers,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_window_rows import (  # noqa: E501
    dflash2_window_rows,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

pytestmark = on_sm(100, 103, reason="the TRTLLM-Gen sparse MLA decode runs on SM100/SM103")

#: Bands in units of 2**-8 of each row's largest output (the scale `assert_within_ulp` uses).
#: Both sides round the output to BF16 once; the kernel also rounds the probabilities to BF16
#: (2**-9 relative each) before P.V, so its output lands on a neighbouring BF16 value of the
#: reference's about as often as not. Measured on the cells: at most 2 units per element and
#: 0.55 .. 0.82 units in relative RMS -- the size of one output rounding.
_BF16_BAND = dict(element_ulp=4.0, rms_ulp=1.0)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch, width, heads, page_size = spec["batch"], spec["width"], spec["heads"], spec["page_size"]
    dim = spec["latent"] + spec["rope"]
    ctx_len = torch.randint(0, 3000, (batch,), generator=g, device="cuda")
    ctx_len[0] = 9
    pages = (3000 + width) // page_size + 2
    num_pages = 2 * batch * pages
    tables = torch.randperm(num_pages, generator=g, device="cuda")[: batch * pages]
    tables = tables.view(batch, pages)
    rows, lengths = dflash2_window_rows(
        ctx_len, tables, width, page_size, spec["window"], spec["window"]
    )
    base = 0.5 * torch.randn(num_pages, spec["layers"], page_size, dim, generator=g, device="cuda")
    pool = base[:, spec["layers"] - 1].bfloat16()
    query = (0.5 * torch.randn(rows.shape[0], heads, dim, generator=g, device="cuda")).bfloat16()
    workspace, counter = dflash2_window_attention_buffers(rows.shape[0], heads, query.dtype, "cuda")
    return query, pool, rows, lengths, spec["scale"], spec["latent"], workspace, counter


CELLS = sample_cells(op.CELLS, size_key="batch")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    out = op(*args)
    with true_fp32_matmul():
        assert_within_ulp(out, op.reference(*args), **_BF16_BAND)
