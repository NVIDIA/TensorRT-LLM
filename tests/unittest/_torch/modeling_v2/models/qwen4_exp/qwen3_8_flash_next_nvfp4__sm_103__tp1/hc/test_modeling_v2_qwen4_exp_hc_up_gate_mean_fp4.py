# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_hc_up_gate_mean_fp4 entry: `mixed` in the band
`qwen4_exp_hc_up_gate_mean`'s test uses, the NVFP4 outputs against the
native-torch quantization of the returned `mixed` outside the reciprocal tie
window, the global-scale guard, graph capture."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    Fp4Quantize,
    fp4_quantize,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_gated_residual import (  # noqa: E501
    gate_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_hc_up_gate_mean_fp4 import (  # noqa: E501
    qwen4_exp_hc_up_gate_mean_fp4 as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_hc_up_gate_mean_fp4 requires a CUDA device"

DEV = "cuda"


def nvfp4_reference(x, gs):
    """`fp4_quantize`'s native-torch NVFP4 of `x` `[T, K]` with linear scales:
    `(packed [T, K/2] uint8, sf [T*K/16] uint8)`."""
    return fp4_quantize.reference(x, gs, 16, False, False)


def nvfp4_near_tie(x, gs):
    """`[T, K]` bool: elements within `fp4_quantize`'s tie window of an e2m1 midpoint,
    where the kernels' approximate reciprocals may round either way."""
    return Fp4Quantize.near_tie(fp4_quantize.scaled(x, gs, 16))


def _unpack(xq: torch.Tensor) -> torch.Tensor:
    return torch.stack([xq & 0xF, xq >> 4], -1).flatten(-2)


def _check_quant(mixed, xq, xsf, gs) -> None:
    """NVFP4 outputs against the native-torch quantization of the `mixed` this
    call returned: scale bytes equal, e2m1 codes equal except inside the
    reciprocal tie window."""
    want_q, want_sf = nvfp4_reference(mixed, gs)
    assert xq.dtype == torch.uint8 and xq.shape == want_q.shape and xq.is_contiguous()
    assert xsf.dtype == torch.uint8 and xsf.shape == want_sf.shape
    assert torch.equal(xsf, want_sf), f"{(xsf != want_sf).sum().item()} scale bytes differ"
    diff = _unpack(xq) != _unpack(want_q)
    stray = diff & ~nvfp4_near_tie(mixed, gs)
    assert not stray.any(), f"{stray.sum().item()} e2m1 codes differ outside the tie window"


def _compare(out, ref, gs) -> None:
    """`mixed`: gate_ratio <= 1, the band `qwen4_exp_hc_up_gate_mean` is
    certified in. Quantization: exact on the returned `mixed`."""
    (mixed, xq, xsf), (ref_mixed, _, _) = out, ref
    assert mixed.dtype == ref_mixed.dtype and mixed.shape == ref_mixed.shape
    assert mixed.is_contiguous()
    assert gate_ratio(mixed, ref_mixed) <= 1.0, f"{gate_ratio(mixed, ref_mixed):.4g}x the band"
    _check_quant(mixed, xq, xsf, gs)


def _build(spec, seed):
    hc, hidden, rank = spec.get("hc_count", 4), spec.get("hidden", 2560), spec.get("rank", 320)
    g = torch.Generator(device=DEV).manual_seed(seed)
    rows, width = spec["rows"], hc * hidden
    # This checkpoint's statistics: low = silu(.) of the bottleneck (|x| < ~3), the up
    # weight ~N(0, 0.068) (gate logits within about +-8), the normed stream ~+-30.
    low = torch.nn.functional.silu(torch.randn(rows, rank, generator=g, device=DEV) * 1.5).to(
        torch.bfloat16
    )
    up_w = (torch.randn(width, rank, generator=g, device=DEV) * 0.068).to(torch.bfloat16)
    normed = (torch.randn(rows, width, generator=g, device=DEV) * 6).to(torch.bfloat16)
    # A calibrated activation global scale: 448 * 6 / amax over a typical batch.
    gs = torch.tensor([448.0 * 6.0 / 24.0], device=DEV, dtype=torch.float32)
    return low, up_w.t(), normed, hc, gs


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % (2**31)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    certifying(op)
    args = _build(cell.spec, seed=_seed(cell))
    out = op(*args)
    assert out[0].shape == (cell.spec["rows"], cell.spec.get("hidden", 2560))
    _compare(out, op.reference(*args), args[4])


def test_guards() -> None:
    """The kernel loads `global_scale[0]` only; a per-row scale is refused."""
    low, up, normed, c, gs = _build(dict(rows=4), seed=4)
    with validating(op), pytest.raises(AssertionError):
        op(low, up, normed, c, torch.cat([gs, gs]))


def test_cuda_graph_capture_and_replay() -> None:
    low, up, normed, c, gs = _build(dict(rows=8), seed=5)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(low, up, normed, c, gs)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = op(low, up, normed, c, gs)
    fresh = _build(dict(rows=8), seed=6)
    low.copy_(fresh[0])
    normed.copy_(fresh[2])
    gs.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    want = op(fresh[0], up, fresh[2], c, gs)
    for got, exp in zip(g_out, want):
        assert torch.equal(got, exp)
