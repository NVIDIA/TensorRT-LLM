# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the norm/m3_embedding_firstnorm entry: the decode embedding gather + the first layer's RMSNorm.

The table is the model's full bf16 [200064, 6144] embedding shape (random values at the checkpoint's scale), eps the
checkpoint's 1e-6.
"""

import functools
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.norm.m3_embedding_firstnorm import (  # noqa: E501
    VOCAB,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.norm.m3_embedding_firstnorm import (  # noqa: E501
    m3_embedding_firstnorm as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_embedding_firstnorm requires a CUDA device"

H = 6144
EPS = 1e-6


@functools.lru_cache(maxsize=None)
def _operands():
    g = torch.Generator(device="cuda").manual_seed(20261013)
    table = (0.02 * torch.randn(VOCAB, H, generator=g, device="cuda")).to(torch.bfloat16)
    weight = (1.0 + 0.1 * torch.randn(H, generator=g, device="cuda")).to(torch.bfloat16)
    return table, weight


def _buffers(rows):
    return (torch.empty(rows, H, dtype=torch.bfloat16, device="cuda"),
            torch.empty(rows, H, dtype=torch.bfloat16, device="cuda"))  # fmt: skip


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    rows, id_dtype = cell.spec["rows"], cell.spec["id_dtype"]
    table, weight = _operands()
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    ids = torch.randint(0, VOCAB, (rows,), generator=g, device="cuda").to(id_dtype)
    raw, out = _buffers(rows)
    with validating(op):
        normed, residual = op(ids, table, weight, EPS, raw, out)
    assert normed.data_ptr() == out.data_ptr() and residual.data_ptr() == raw.data_ptr()
    ref_normed, ref_raw = op.reference(ids, table, weight, EPS, raw, out)
    # The gather is a copy: exact.
    assert torch.equal(raw.view(torch.int16), ref_raw.view(torch.int16))
    # The norm: the kernel's fp32 sum of squares (warp butterflies, then across 4 warps) and fast rsqrt against an
    # fp64 reference, then x * rstd * w rounded once to bf16. The fp32 side is ~1e-6 relative off, so the two roundings
    # disagree only next to a bf16 rounding boundary: one ulp at most per element, a small fraction in aggregate.
    # Measured over these cells (3 seeds each, GB300): at most 0.51 ulp element, 0.026 ulp RMS.
    assert_within_ulp(out, ref_normed, element_ulp=1.0, rms_ulp=0.1)


def test_ids_outside_the_table_give_zero_rows() -> None:
    """Documented, not refused: an id outside [0, V) gathers a zero row whose norm is zero."""
    table, weight = _operands()
    ids = torch.tensor([-1, 5, VOCAB], dtype=torch.int64, device="cuda")
    raw, out = _buffers(3)
    op(ids, table, weight, EPS, raw, out)
    assert not raw[0].any() and not raw[2].any() and not out[0].any() and not out[2].any()
    assert torch.equal(raw[1], table[5])
