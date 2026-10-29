# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_select_blocks entry of the MiniMax-M3 tp8 target: MSA top-16 block selection with the forced
init / local blocks and per-row valid-block masks."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.attention.m3_select_blocks import (  # noqa: E501
    m3_select_blocks as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_select_blocks requires a CUDA device"

TOPK, INIT, LOCAL = 16, 0, 1


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    n_blocks, total_q = spec["n_blocks"], spec["total_q"]
    # Distinct scores (a permutation) so the selection has no ties to break.
    scores = (
        torch.randperm(n_blocks * total_q, generator=g, device="cuda")
        .float()
        .view(1, n_blocks, total_q)
    )
    if spec["mixed"]:
        n_valid = torch.randint(
            17, n_blocks + 1, (total_q,), generator=g, device="cuda", dtype=torch.int32
        )
    else:
        n_valid = torch.full((total_q,), n_blocks, dtype=torch.int32, device="cuda")
    with validating(op):
        out = op(scores, TOPK, n_valid, INIT, LOCAL, True)
    ref = op.reference(scores, TOPK, n_valid, INIT, LOCAL, True)
    assert out.shape == ref.shape
    assert torch.equal(out, ref)
    assert out.permute(1, 0, 2).is_contiguous(), "head-major backing"
