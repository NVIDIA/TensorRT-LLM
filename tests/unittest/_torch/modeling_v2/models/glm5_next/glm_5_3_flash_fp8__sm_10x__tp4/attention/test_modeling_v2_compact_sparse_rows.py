# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the glm5_next fp8 tp4 entry compact_sparse_rows."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_fp8__sm_10x__tp4.ops.attention.compact_sparse_rows import (  # noqa: E501
    compact_sparse_rows,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 100, reason="the glm5_next fp8 tp4 entries are certified on sm_100"
)

DEVICE = "cuda"
SELECT_K, KPOOL = 512, 4


def _expanded(spec: dict, seed: int) -> torch.Tensor:
    """Rows shaped like ``kpool_expand``'s output, from each row's visible length (the cell's
    ``kv`` for the last row, one less per row before it): the members of its selected pools,
    its tail, ``-1`` padding. 5% of the entries are knocked out to ``-1`` so that every row has
    holes to close, and the last row selects nothing."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    rows, width = spec["rows"], spec["width"]
    visible = (spec["kv"] - torch.arange(rows, device=DEVICE).flip(0)).clamp(min=0)
    pools = (visible // KPOOL).clamp(max=SELECT_K)
    col = torch.arange(width, device=DEVICE)
    members = (col < SELECT_K * KPOOL) & (col // KPOOL < pools[:, None])
    tail = col - SELECT_K * KPOOL
    tail = (tail >= 0) & (tail < (visible % KPOOL)[:, None])
    ids = torch.randint(0, 1 << 24, (rows, width), generator=g, device=DEVICE, dtype=torch.int32)
    holes = torch.rand(rows, width, generator=g, device=DEVICE) < 0.05
    indices = torch.where((members | tail) & ~holes, ids, -1)
    indices[-1] = -1
    return indices


@pytest.mark.parametrize(
    "cell", compact_sparse_rows.CELLS, ids=[c.why[:44] for c in compact_sparse_rows.CELLS]
)
def test_compact_matches_reference(cell, certifying):  # noqa: F811
    """Packed ids, lengths and the nonempty mask equal the reference's exactly: the op moves
    integers, so there is no tolerance. The last row, which selects nothing, gets the dummy
    id 0, length 1 and nonempty False."""
    certifying(compact_sparse_rows)
    indices = _expanded(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    out = compact_sparse_rows(indices)
    ref = compact_sparse_rows.reference(indices)
    for got, want in zip(out, ref):
        assert got.dtype == want.dtype and got.shape == want.shape
        assert torch.equal(got, want)
    packed, lengths, nonempty = out
    assert int(packed[-1, 0]) == 0 and int(lengths[-1]) == 1 and not bool(nonempty[-1])


def test_guard_refuses_column_strided_ids():
    """A column-strided view is read as if its columns were adjacent: the op would pack
    entries of the underlying buffer the view does not hold."""
    buf = torch.randint(-1, 100, (4, 2 * 2112), device=DEVICE, dtype=torch.int32)
    with pytest.raises(AssertionError, match="unit-stride"):
        with validating(compact_sparse_rows):
            compact_sparse_rows(buf[:, ::2])
