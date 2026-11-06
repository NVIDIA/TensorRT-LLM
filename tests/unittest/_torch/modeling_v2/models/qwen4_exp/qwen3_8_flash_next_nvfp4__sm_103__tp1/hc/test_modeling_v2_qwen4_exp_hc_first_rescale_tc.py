# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_first_rescale_tc entry: `low` and `inject` in
the gate band (`qwen4_exp_gated_residual`'s) of the float64 projection of the
once-rounded normed stream, the stack-final form, the weight-layout guard,
graph capture.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_gated_residual import (  # noqa: E501
    gate_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_first_rescale_tc import (  # noqa: E501
    qwen4_exp_hc_first_rescale_tc as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_first_rescale_tc requires a CUDA device"

DEV = "cuda"
HC_COUNT = 4
HIDDEN = 2560
RANK = 320
EPS = 1e-6
#: The padded first weight's rows (`_fuse_hc_first_projections`: 324 rounded up to 128).
FIRST_ROWS = 384
# This checkpoint's grouped-norm weight statistics (layer-0 hyper-connection norm).
GAMMA_RMS = 0.477
GAMMA_ABSMAX = 6.625


def _compare(out, ref) -> None:
    """gate_ratio <= 1 for `low` and `inject`: the band the mixer's outputs are
    certified in (`rtol = eps`, `atol = eps * max|ref|`); the fp32 logits are a
    10240-term tensor-core sum against the float64 one.
    """
    for got, want in zip(out, ref):
        assert (got is None) == (want is None)
        if got is None:
            continue
        assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
        if got.numel():
            assert gate_ratio(got, want) <= 1.0, f"{gate_ratio(got, want):.4g}x the band"


def _build(spec, seed):
    rows, width, dtype = spec["rows"], HC_COUNT * HIDDEN, torch.bfloat16
    g = torch.Generator(device=DEV).manual_seed(seed)
    hc_out = (torch.randn(rows, width, generator=g, device=DEV) * 3.0).to(dtype)
    x = hc_out.double().unflatten(-1, (HC_COUNT, HIDDEN))
    rstd = torch.rsqrt(x.pow(2).mean(-1) + EPS).float().contiguous()
    gamma = torch.randn(width, generator=g, device=DEV) * GAMMA_RMS
    gamma[0], gamma[width // 2] = GAMMA_ABSMAX, -GAMMA_ABSMAX
    # `[down; block_inject; 0]`, logits up to about +-60 (this checkpoint's reach |67|).
    first = torch.zeros(FIRST_ROWS, width, device=DEV)
    first[: RANK + HC_COUNT] = torch.randn(RANK + HC_COUNT, width, generator=g, device=DEV) * 0.15
    return (
        hc_out,
        rstd,
        gamma.to(dtype),
        first.to(dtype).t(),
        HC_COUNT,
        RANK,
        bool(spec.get("inject", True)),
    )


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % (2**31)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell))
    low, inject = op(*args)
    assert low.shape == (cell.spec["rows"], RANK)
    assert (inject is None) == (not args[6])
    _compare((low, inject), op.reference(*args))


def test_stack_final_low_is_the_injecting_form_bitwise() -> None:
    args = _build(dict(rows=300), seed=7)
    low, inject = op(*args[:6], True)
    low_f, none = op(*args[:6], False)
    assert none is None and inject.shape == (300, HC_COUNT)
    assert torch.equal(low, low_f)


def test_guard_refuses_a_contiguous_first() -> None:
    """The kernel streams the `[N, C*H]` weight row-major: `first` must be its
    transposed view, not a `[C*H, N]` copy.
    """
    args = list(_build(dict(rows=4), seed=4))
    args[3] = args[3].contiguous()
    with validating(op), pytest.raises(AssertionError, match="transposed"):
        op(*args)


def test_cuda_graph_capture_and_replay() -> None:
    args = _build(dict(rows=8), seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(*args)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_low, g_inject = op(*args)
    fresh = _build(dict(rows=8), seed=6)
    args[0].copy_(fresh[0])
    args[1].copy_(fresh[1])
    graph.replay()
    torch.cuda.synchronize()
    e_low, e_inject = op(*args)
    assert torch.equal(g_low, e_low) and torch.equal(g_inject, e_inject)
