# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the gated_ngram_injection entry: the cells (the sub-floor and
zero dots are a cell), the leading-dim guard and CUDA-graph replay."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.gated_ngram_injection import (  # noqa: E501
    gate_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.ple.gated_ngram_injection import (  # noqa: E501
    gated_ngram_injection as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "gated_ngram_injection requires a CUDA device"


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """16-bit: `rtol = eps`, `atol` one subnormal spacing (`gate_ratio <= 1`)
    -- half torch's bf16 `rtol`. Nothing cancels (each element is a sigmoid
    times one `value` element), so the budget is the final rounding; with
    the reference also rounded once, op and reference agree or sit one ulp
    apart. One ulp is at most `eps * |ref|`, so this band always admits it
    and never two. Measured at 2048 tokens: 0.992 (a one-ulp element just
    above a binade boundary), eight wrong readings at 105x..1115x. The
    `atol` is for Gaussian `value` elements that land subnormal (measured
    on fp16).

    fp32: torch's default band. The 2560-term signed dot is accumulated at
    the width it is reported in, so its conditioning, not the cast, sets
    the error; a custom pair would be fitted to a result.
    """
    assert out.dtype == ref.dtype, (out.dtype, ref.dtype)
    assert out.shape == ref.shape, (out.shape, ref.shape)
    assert out.is_contiguous()
    if ref.dtype == torch.float32:
        torch.testing.assert_close(out, ref)
        return
    ratio = gate_ratio(out, ref)
    assert ratio <= 1.0, f"{ratio:.4g}x the band"


DEV = "cuda"
HC_COUNT = 4
HIDDEN = 2560
WIDTH = HC_COUNT * HIDDEN
RMS_NORM_EPS = 1e-6
CLAMP_MIN = 1e-6

# This checkpoint's PLE tensors: norm_key / norm_query gammas (element RMS
# 0.1358, extremes 0.863 and -0.326), and the value's scale derived as
# sqrt(2560) * 0.015295 (value_proj RMS) * 0.00747 (dequantized table RMS).
# The entry is exactly linear in `value`, so its absolute scale decides nothing.
CKPT_NORM_GAMMA_RMS = 0.1358
CKPT_NORM_GAMMA_MAX = 0.863281
CKPT_NORM_GAMMA_MIN = -0.326172
CKPT_VALUE_RMS = 0.0058


def _gemma_grouped_rmsnorm(x, gamma, hc_count):
    """Where key/query come from -- here only for their distribution."""
    hidden = x.shape[-1] // hc_count
    xf = x.float().reshape(*x.shape[:-1], hc_count, hidden)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_NORM_EPS)
    return (xf.flatten(-2) * (1.0 + gamma.float())).to(x.dtype)


def _make_operands(seed, dtype, leading, hc_count=HC_COUNT, value_dtype=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    width = hc_count * HIDDEN

    def normal(*dims, scale, dt=dtype):
        return (torch.randn(*dims, generator=g, device=DEV) * scale).to(dt)

    gammas = []
    for _ in range(2):
        gamma = normal(width, scale=CKPT_NORM_GAMMA_RMS)
        gamma[0] = CKPT_NORM_GAMMA_MAX
        gamma[width // 2] = CKPT_NORM_GAMMA_MIN
        gammas.append(gamma)
    key = _gemma_grouped_rmsnorm(normal(*leading, width, scale=1.0), gammas[0], hc_count)
    query = _gemma_grouped_rmsnorm(normal(*leading, width, scale=1.0), gammas[1], hc_count)
    value = normal(*leading, HIDDEN, scale=CKPT_VALUE_RMS, dt=value_dtype or dtype)
    return key, query, value


def _single_channel_operands(dots, value_dtype):
    """One nonzero channel per stream: each dot is exactly `dots[s]`."""
    hc_count = len(dots)
    key = torch.zeros(8, hc_count * HIDDEN, dtype=torch.bfloat16, device=DEV)
    query = torch.zeros_like(key)
    for stream, dot in enumerate(dots):
        key[:, stream * HIDDEN] = 1.0
        query[:, stream * HIDDEN] = dot
    value = torch.full((8, HIDDEN), 0.5, dtype=value_dtype, device=DEV)
    return key, query, value


def _build(spec, seed):
    if spec.get("kind") == "single_channel":
        dots = spec["dots"]
        key, query, value = _single_channel_operands(dots, spec["value_dtype"])
        return key, query, value, len(dots), spec["clamp_min"]
    key, query, value = _make_operands(
        seed, torch.bfloat16, spec["leading"], value_dtype=spec.get("value_dtype")
    )
    return key, query, value, HC_COUNT, spec.get("clamp_min", CLAMP_MIN)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()))
    before = [a.clone() for a in args[:3]]
    out = op(*args)
    key, _, value, hc_count, _ = args
    assert out.shape == (*key.shape[:-1], hc_count * value.shape[-1])
    assert out.dtype == value.dtype
    for a, b in zip(args[:3], before):
        assert torch.equal(a, b), "the operands are read-only"
    _compare(out, op.reference(*args))


def test_guard_refuses_unequal_leading_dims() -> None:
    """Unguarded, one query (or value) row is broadcast over every key row."""
    key, query, value = _make_operands(60, torch.bfloat16, (64,))
    unguarded = op(key, query[:1], value, HC_COUNT)
    assert unguarded.shape == (64, WIDTH), "the hazard this guard covers is gone"
    assert not torch.equal(unguarded, op.reference(key, query, value, HC_COUNT))
    assert op(key, query, value[:1], HC_COUNT).shape == (64, WIDTH)

    with pytest.raises(AssertionError, match="broadcasts silently"):
        with validating(op):
            op(key, query[:1], value, HC_COUNT)
    with pytest.raises(AssertionError, match="broadcasts silently"):
        with validating(op):
            op(key, query, value[:1], HC_COUNT)


def test_cuda_graph_capture_and_replay() -> None:
    key, query, value = _make_operands(100, torch.bfloat16, (8,))
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(key, query, value, HC_COUNT, CLAMP_MIN)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(key, query, value, HC_COUNT, CLAMP_MIN)

    fresh = _make_operands(101, torch.bfloat16, (8,))
    for static, new in zip((key, query, value), fresh):
        static.copy_(new)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op(*fresh, HC_COUNT, CLAMP_MIN))
