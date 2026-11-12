# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the nvfp4_w4a16_gemv entry.

Cells against the exact-dequantization fp32 reference on decode copies built by the
recipe (`fp8_to_nvfp4_values` + `gemv_layout`) from FP8 weights with outlier input
channels, activations with outlier columns. Beyond the cells: a PDL producer that
writes `x` late (weight loads are issued before the wait), graph replay against
eager, and the alignment guard.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.gemm.nvfp4_w4a16_gemv import (  # noqa: E501
    nvfp4_w4a16_gemv as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.quantization import (  # noqa: E501
    _nvfp4_recipes as recipes,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import capture, late_producer, seed_of  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """Accumulation order only: E2M1 x E4M3 x BF16 products are exact in fp32 on both
    sides, the kernel sums per-16-block partials, a warp tree and the warps in order,
    torch its own order; both multiply by the power-of-two global scale and round once.
    That moves an output only at a BF16 rounding midpoint: a one-ulp flip is 2 units of
    the row maximum's 2**-8, so 4 units element-wise and 2 of relative RMS, as for the
    other fp32-accumulating GEMVs of this tree."""
    assert out.dtype == torch.bfloat16 and out.is_contiguous()
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=2.0)


def _copy(n: int, k: int, seed: int) -> recipes.GemvNvfp4Weight:
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randn(n, k, generator=g, device=DEV) * 0.02
    w = w * (torch.rand(n, 1, generator=g, device=DEV) * 1.5 + 0.25)
    w[:, 3::211] *= 8.0  # a few outlier input channels
    s = (w.abs().amax() / 448.0).float()
    w8 = (w / s).to(torch.float8_e4m3fn)
    return recipes.gemv_layout(recipes.fp8_to_nvfp4_values(w8, s.reshape(())))


def _x(m: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(m, k, generator=g, device=DEV) * 0.5
    x[:, ::97] *= 20.0  # outlier activation columns
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    c = _copy(s["n"], s["k"], seed=s["n"] + s["k"])
    x = _x(s["m"], s["k"], seed_of(cell.why))
    out = op(x, c.weight, c.weight_sf, c.global_scale)
    assert out.shape == (s["m"], s["n"])
    _compare(out, op.reference(x, c.weight, c.weight_sf, c.global_scale))


@pytest.mark.parametrize("n,k,m", [(5120, 6144, 1), (16384, 5120, 3)], ids=["out_proj-1", "qkvz-3"])
def test_input_written_late_by_a_pdl_producer(n, k, m) -> None:
    """Weight and scale loads precede `griddepcontrol.wait`; `x` must not."""
    c = _copy(n, k, seed=n)
    src = _x(m, k, seed=m)
    want = op(src, c.weight, c.weight_sf, c.global_scale).clone()
    buf = torch.empty_like(src)

    def step():
        buf.fill_(float("nan"))
        late_producer(src, buf)
        return op(buf, c.weight, c.weight_sf, c.global_scale)

    for _ in range(5):
        assert torch.equal(step(), want)
    graph, out = capture(step)
    for _ in range(10):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, want)


def test_graph_replay_matches_eager() -> None:
    c = _copy(5120, 6144, seed=1)
    x = _x(3, 6144, seed=2)
    graph, out = capture(lambda: op(x, c.weight, c.weight_sf, c.global_scale))
    for trial in range(3):
        x.copy_(_x(3, 6144, seed=10 + trial))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, op(x, c.weight, c.weight_sf, c.global_scale))


def test_guard_refuses_a_misaligned_activation() -> None:
    c = _copy(5120, 6144, seed=3)
    x = _x(1, 6144 + 8, seed=4)[:, 1:-7].contiguous()  # aligned copy...
    base = torch.empty(6144 + 1, dtype=torch.bfloat16, device=DEV)
    misaligned = base[1:].view(1, 6144)  # ...and a 2-byte-offset view of the same values
    misaligned.copy_(x)
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 16 != 0
    with validating(op), pytest.raises(AssertionError, match="16-byte aligned"):
        op(misaligned, c.weight, c.weight_sf, c.global_scale)
