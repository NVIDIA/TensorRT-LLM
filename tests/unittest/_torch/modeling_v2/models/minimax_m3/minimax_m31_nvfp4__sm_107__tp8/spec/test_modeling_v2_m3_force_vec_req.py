# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_force_vec_req entry of the MiniMax-M3 tp8 target: the per-request fractional forced-acceptance
draw (one draw per request at its own key, no shared counter)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8.ops.spec.m3_force_vec import (  # noqa: E501
    m3_force_vec_req as op,
)
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_force_vec_req requires a CUDA device"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(7)
    pool = torch.rand(_FORCE_ACCEPT_RNG_POOL_SIZE, generator=g, device="cuda")
    # Each request's key is its KV length at the verify: long, distinct, arbitrary.
    keys = torch.randint(
        1, 1 << 20, (spec["batch"],), generator=g, device="cuda", dtype=torch.int64
    ).to(torch.int32)
    out = torch.empty(spec["batch"], dtype=torch.int32, device="cuda")
    args = (
        pool,
        keys,
        out,
        spec["batch"],
        2,
        4,
        spec["frac"],
        _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
        _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
        spec["block"],
    )
    ref = op.reference(*args)
    with validating(op):
        op(*args)
    assert torch.equal(out, ref)
