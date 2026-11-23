# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the embedding/m31_embedding entry of the MiniMax-M3.1-Flash sm_107 targets: an exact gather with
zero rows outside the table."""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.embedding.m31_embedding import (  # noqa: E501
    m3_embedding as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

VOCAB, HIDDEN = 4096, 6144
# The targets share most cells; each distinct one runs once.
CELLS = list({repr(c): c for c in op.CELLS}.values())


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    table = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    ids = torch.randint(
        0, VOCAB, (spec["rows"],), generator=g, device="cuda", dtype=spec["id_dtype"]
    )
    if spec["pad"]:
        ids[1::3] = -1
    with validating(op):
        out = op(ids, table)
    assert torch.equal(out, op.reference(ids, table))
    if spec["pad"]:
        assert not out[1::3].any(), "padded ids must embed to zero rows"
