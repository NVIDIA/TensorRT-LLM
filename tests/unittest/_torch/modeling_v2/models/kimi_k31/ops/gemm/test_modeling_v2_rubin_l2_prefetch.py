# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the rubin_l2_prefetch entry: an L2 prefetch changes no byte, on one of the entry's cells."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.rubin_l2_prefetch import (  # noqa: E501
    rubin_l2_prefetch as op,
)

__extra_import_path__ = [".", "../../../..", "../.."]
from _k31_utils import sample_cells  # noqa: E402 — needs the path declared above
from _rubin_gemm_util import generator, requires_sm107  # noqa: E402 — needs the path declared above
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = requires_sm107

CELLS = sample_cells(op.CELLS)[:1]


def _ranges(tensors):
    """The ``(address, bytes)`` range of each tensor."""
    return tuple((t.data_ptr(), t.numel() * t.element_size()) for t in tensors)


def _buffers(spec, gen):
    return [
        torch.randint(0, 256, (size,), generator=gen, device="cuda", dtype=torch.uint8)
        for size in spec["sizes"]
    ]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    buffers = _buffers(cell.spec, generator(cell.why))
    before = [b.clone() for b in buffers]
    ranges = _ranges(buffers)
    assert op(ranges, max_ctas=cell.spec["max_ctas"]) is None
    torch.cuda.synchronize()
    assert op.reference(ranges, max_ctas=cell.spec["max_ctas"]) is None
    assert all(torch.equal(b, a) for b, a in zip(buffers, before))
