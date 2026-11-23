# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_router_tc entry of the MiniMax-M3 tp4 target: the TF32 tensor-core router of the unfused MoE's
9..16-row pure generation steps (the arithmetic the built-in gate takes at these rows)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3 import (  # noqa: F401
        minimax_m3_1_flash_nvfp4__sm_107__tp4,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

__extra_import_path__ = ["../../../..", "."]
from _m3_router import _build, _compare_logits, _seed, tc  # noqa: E402
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)


@pytest.mark.parametrize("cell", tc.CELLS_BY_TP[4], ids=[c.why[:48] for c in tc.CELLS_BY_TP[4]])
def test_tc_certified_cells(cell) -> None:
    spec = cell.spec
    x, w = _build(spec["m"], 128, spec["k"], _seed(cell))
    with validating(tc):
        outs = tc(x, w, None, False)
    ref = tc.reference(x, w, None, False)
    assert len(outs) == len(ref) == 1
    _compare_logits(outs[0], ref[0])


def test_tc_is_the_built_in_gates_router_call() -> None:
    """The target's entry is the op the built-in gate calls at these rows, with the same arguments: the same logits
    bit for bit."""
    x, w = _build(14, 128, 6144, 7)
    ours = tc(x, w, None, True)[0]
    gate = torch.ops.trtllm.minimax_m3_router_tc(x, w, None, True)[0]
    assert torch.equal(ours.view(torch.int32), gate.view(torch.int32))
