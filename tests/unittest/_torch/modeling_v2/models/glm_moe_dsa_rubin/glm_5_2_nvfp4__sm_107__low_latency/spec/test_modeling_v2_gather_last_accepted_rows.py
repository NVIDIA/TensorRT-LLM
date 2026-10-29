# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm_gather_last_accepted_rows entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.spec.gather_last_accepted_rows import (  # noqa: E501
    glm_gather_last_accepted_rows as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — needs the path declared above

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's ops are certified on sm_107"
)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    requests, per_request, cols = spec["requests"], spec["rows_per_request"], spec["cols"]
    rows = torch.randn(requests * per_request, cols, generator=g, device="cuda").to(torch.bfloat16)
    second = None
    if spec["second"]:
        second = torch.randn(requests * per_request, cols, generator=g, device="cuda").to(
            torch.bfloat16
        )
    order = spec.get("batch_order", tuple(range(requests)))
    batch_indices = torch.tensor(order, dtype=torch.int32, device="cuda")
    num_accepted = torch.tensor(spec["accepted"], dtype=torch.int32, device="cuda")
    return rows, batch_indices, num_accepted, per_request, 0, second


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    """A gather moves bytes: the op and the reference agree bit for bit."""
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    out = op(*args)
    ref = op.reference(*args)
    assert len(out) == len(ref) == (3 if cell.spec["second"] else 2)
    for got, expected in zip(out, ref):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_row_offset_shifts_the_selection() -> None:
    """Rows after `row_offset` context tokens: the index moves by the offset, the gathers with it."""
    rows, batch_indices, num_accepted, per_request, _, second = _build(op.CELLS[0].spec, seed=3)
    padded = torch.cat([torch.zeros(5, rows.shape[1], dtype=rows.dtype, device="cuda"), rows])
    padded_second = torch.cat([torch.zeros_like(padded[:5]), second])
    with validating(op):
        out = op(padded, batch_indices, num_accepted, per_request, 5, padded_second)
    ref = op.reference(rows, batch_indices, num_accepted, per_request, 0, second)
    torch.testing.assert_close(out[0], ref[0] + 5, rtol=0, atol=0)
    torch.testing.assert_close(out[1], ref[1], rtol=0, atol=0)
    torch.testing.assert_close(out[2], ref[2], rtol=0, atol=0)


def test_guard_refuses_a_count_of_zero() -> None:
    """A request with no accepted row would select the row before its own."""
    rows, batch_indices, _, per_request, offset, second = _build(op.CELLS[0].spec, seed=4)
    zero = torch.zeros(1, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match=r"\[1, rows_per_request\]"):
        with validating(op):
            op(rows, batch_indices, zero, per_request, offset, second)


def test_guard_refuses_a_wider_second() -> None:
    """The kernel copies the columns of `rows`: a wider `second` keeps garbage in its tail."""
    rows, batch_indices, num_accepted, per_request, offset, _ = _build(op.CELLS[0].spec, seed=5)
    wide = torch.zeros(rows.shape[0], rows.shape[1] + 8, dtype=rows.dtype, device="cuda")
    with pytest.raises(AssertionError, match="columns of rows"):
        with validating(op):
            op(rows, batch_indices, num_accepted, per_request, offset, wide)


def test_guard_refuses_a_strided_count_vector() -> None:
    """The kernel reads the counts as a dense vector."""
    rows, _, _, per_request, offset, second = _build(op.CELLS[2].spec, seed=6)
    batch_indices = torch.arange(4, dtype=torch.int32, device="cuda")
    strided = torch.ones(8, dtype=torch.int32, device="cuda")[::2]
    with pytest.raises(AssertionError, match="contiguous vector"):
        with validating(op):
            op(rows, batch_indices, strided, per_request, offset, second)
