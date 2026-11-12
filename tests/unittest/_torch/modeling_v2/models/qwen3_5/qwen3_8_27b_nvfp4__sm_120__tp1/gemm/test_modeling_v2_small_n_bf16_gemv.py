# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the small_n_bf16_gemv entry.

Cells against the fp32 reference (TF32 off) with activations at three scales and a
few outlier channels. Beyond the cells: CUDA-graph replay (and a fork onto a side
stream, as the decode step runs in_proj_ba beside in_proj_qkvz) bitwise equal to
eager, and the layout guard.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm.small_n_bf16_gemv import (  # noqa: E501
    small_n_bf16_gemv as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import capture, seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Accumulation order only (BF16 x BF16 exact in fp32, per-thread partials then a
    fixed tree against torch's order): one-ulp flips at BF16 midpoints, 2 units of the
    row maximum's 2**-8 each; 4 units element-wise, 2 of RMS, as `low_m_bf16_mm`."""
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _build(m: int, n: int, k: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    scale = (0.5, 1.0, 4.0)[seed % 3]
    x = torch.randn(m, k, generator=g, device=DEV) * scale
    x[:, :16] *= 30.0  # outlier channels
    w = torch.randn(n, k, generator=g, device=DEV) * 0.02
    return x.to(torch.bfloat16), w.to(torch.bfloat16)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    x, w = _build(s["m"], s["n"], s["k"], seed_of(cell.why))
    out = op(x, w)
    assert out.shape == (s["m"], s["n"])
    _compare(out, op.reference(x, w))


def test_forked_graph_replay_matches_eager() -> None:
    """Captured on a side stream forked from and joined to the main one, as the step
    runs it beside in_proj_qkvz; every replay, with new activations, bitwise eager."""
    x, w = _build(16, 96, 5120, seed=1)
    main = torch.cuda.current_stream()
    aux = torch.cuda.Stream()

    def step():
        aux.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(aux):
            y = op(x, w)
        torch.cuda.current_stream().wait_stream(aux)
        return y

    graph, out = capture(step)
    for trial in range(3):
        x.copy_(_build(16, 96, 5120, seed=10 + trial)[0])
        out.fill_(float("nan"))
        graph.replay()
        main.synchronize()
        assert torch.equal(out, op(x, w))


def test_guard_refuses_a_transposed_weight() -> None:
    x, w = _build(4, 96, 5120, seed=2)
    with validating(op), pytest.raises(AssertionError, match="unit inner stride"):
        op(x, w.t().contiguous().t())
