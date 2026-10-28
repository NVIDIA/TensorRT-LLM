# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_force_vec entry of the MiniMax-M3 tp8 target: the fractional forced-acceptance draw."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.spec.m3_force_vec import (  # noqa: E501
    m3_force_vec as op,
)
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
)

__extra_import_path__ = ["../../../.."]
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "m3_force_vec requires a CUDA device"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(5)
    pool = torch.rand(_FORCE_ACCEPT_RNG_POOL_SIZE, generator=g, device="cuda")
    counter = torch.tensor([41], dtype=torch.int64, device="cuda")
    out = torch.empty(spec["batch"], dtype=torch.int32, device="cuda")
    args = (
        pool,
        counter,
        out,
        spec["batch"],
        2,
        4,
        spec["frac"],
        _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
        _FORCE_ACCEPT_RNG_SLOT_STRIDE,
        _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
        spec["block"],
    )
    ref = op.reference(*args)
    with validating(op):
        op(*args)
    assert int(counter.item()) == 42, "the draw bumps the step counter once"
    assert torch.equal(out, ref)
