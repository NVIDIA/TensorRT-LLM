# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_router_tc entry of the MiniMax-M3 tp4 target: the TF32 tensor-core router of the unfused MoE's
9..16-row pure generation steps (the arithmetic the built-in gate takes at these rows)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul

__extra_import_path__ = ["../../../..", "."]
from _m3_router import _build, _compare_logits, _seed, tc  # noqa: E402
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)


@pytest.mark.parametrize("cell", tc.CELLS, ids=[c.why[:48] for c in tc.CELLS])
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


def test_tc_reference_is_tf32_and_the_gate_sees_it() -> None:
    """The tensor-core router rounds the weight to TF32: against a true fp32 product its logits sit far outside
    the band, so the TF32 rounding in ``reference`` is load-bearing rather than decoration."""
    x, w = _build(16, 128, 6144, 5)
    out = tc(x, w, None, False)[0]
    with true_fp32_matmul():
        fp32 = x.float() @ w.t()
    with pytest.raises(AssertionError):
        _compare_logits(out, fp32)


def test_guard_refuses_a_misaligned_activation() -> None:
    """A row view starting 8 bytes into the buffer keeps a legal row stride but not a legal bulk-copy source."""
    x, w = _build(9, 128, 6144, 3)
    flat = torch.empty(9 * 6144 + 4, dtype=torch.bfloat16, device="cuda")
    shifted = flat[4:].view(9, 6144)
    shifted.copy_(x)
    with pytest.raises(AssertionError, match="16-byte aligned"):
        with validating(tc):
            tc(shifted, w, None, False)
