# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_embedding entry of the MiniMax-M3 tp8 target: an exact gather with zero rows outside the
table."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.embedding.m3_embedding import (  # noqa: E501
    m3_embedding as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_embedding requires a CUDA device"

VOCAB, HIDDEN = 4096, 6144


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
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
