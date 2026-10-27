# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_router_tc entry of the MiniMax-M3.1 tp2 target (ROUTER48: the sm_107 TP8 wrapper, cells
2..48 rows): the TF32 tensor-core router of the unfused MoE's generation steps, plain logits and logits + the NVFP4
activation. Every certified cell runs; the K7 c3 / c4 steps (24 / 32 rows) are also driven explicitly in both forms,
since ROUTER48 is what takes them at TP2 (ROUTER_TC_GEN 9..48)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul

__extra_import_path__ = ["../../../..", "."]
from _m3_router import SF_SCALE, _build, _compare_fp4, _compare_logits, _seed, tc  # noqa: E402
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)


def _run(m: int, fp4: bool, seed: int) -> None:
    x, w = _build(m, 128, 6144, seed)
    sf_scale = torch.tensor([SF_SCALE], dtype=torch.float32, device="cuda") if fp4 else None
    with validating(tc):
        outs = tc(x, w, sf_scale, False)
    ref = tc.reference(x, w, sf_scale, False)
    assert len(outs) == len(ref) == (3 if fp4 else 1)
    _compare_logits(outs[0], ref[0])
    if fp4:
        _compare_fp4(outs[1], outs[2], x, sf_scale)


@pytest.mark.parametrize("cell", tc.CELLS, ids=[c.why[:48] for c in tc.CELLS])
def test_tc_certified_cells(cell) -> None:
    _run(cell.spec["m"], cell.spec["fp4"], _seed(cell))


def test_the_cells_certify_every_router48_row() -> None:
    """The target's ROUTER_TC_GEN row range (9..48 with ROUTER48) is inside the wrapper's certified cells."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
        modeling as dispatch,
    )

    assert dispatch.ROUTER48 and dispatch.ROUTER_TC_GEN_MAX_ROWS == 48
    assert dispatch.ROUTER_TC_GEN.keys <= dispatch.ROUTER_TC_GEN.certified()
    assert {24, 32} <= dispatch.ROUTER_TC_GEN.keys


@pytest.mark.parametrize("fp4", [False, True], ids=["logits", "logits_nvfp4"])
@pytest.mark.parametrize("m", [24, 32], ids=["c3_24rows", "c4_32rows"])
def test_tc_k7_c3_c4_rows(m, fp4) -> None:
    for seed in range(3):
        _run(m, fp4, 7000 + 100 * m + seed)


def test_tc_is_the_built_in_gates_router_call() -> None:
    """The target's entry is the op the built-in gate calls at these rows, with the same arguments: the same logits
    bit for bit."""
    for m in (14, 24, 32):
        x, w = _build(m, 128, 6144, 7)
        ours = tc(x, w, None, True)[0]
        gate = torch.ops.trtllm.minimax_m3_router_tc(x, w, None, True)[0]
        assert torch.equal(ours.view(torch.int32), gate.view(torch.int32)), m


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
