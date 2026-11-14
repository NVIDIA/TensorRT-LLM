# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_embedding entry of the MiniMax-M3 tp4 target: an exact gather with zero rows outside the
table."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.embedding.m3_embedding import (  # noqa: E501
        m3_embedding as op,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 TP2 target runs on SM107 only", allow_module_level=True)

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
