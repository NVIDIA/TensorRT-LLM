# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_gated_residual entry (this target's local variant r42).

The thing under test is a formula -- which operand is gated, which axis is
reduced, where each division by `hc_count` sits, the factor on the injection
-- and every mis-reading of it keeps the shapes; the cells' band sits 32x..192x
inside the nearest wrong formula (measured). Beyond the cells: the band refusing
wrong formulas, graph replay and one loud misuse.
"""

import zlib
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.hc.qwen4_exp_gated_residual__r42 import (  # noqa: E501
    gate_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.hc.qwen4_exp_gated_residual__r42 import (  # noqa: E501
    qwen4_exp_gated_residual as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qwen4_exp_gated_residual requires a CUDA device"


def _compare(
    out: Tuple[torch.Tensor, Optional[torch.Tensor]],
    ref: Tuple[torch.Tensor, Optional[torch.Tensor]],
) -> None:
    """Each result within `rtol = eps`, `atol = eps * max|ref|` of the
    activation dtype (`gate_ratio <= 1`).

    The absolute term is not torch's 1e-5: `mixed` is a mean of signed
    products, so its near-zero elements are cancellations whose error is
    set by the whole stream's rounding, and a relative gate on them gates
    nothing. The relative term is half torch's bf16 default. Measured on
    sm_103 at 512 tokens: the correct result at 0.341 (`mixed`) and 0.404
    (`inject`) of this band (0.198 / 0.210 against the unrounded float64,
    the rest is the reference's own rounding), ten wrong formulas at
    32x..192x.
    """
    (mixed, inject), (ref_mixed, ref_inject) = out, ref
    assert (inject is None) == (ref_inject is None), "inject must be None exactly when ref's is"
    pairs = [(mixed, ref_mixed)] + ([] if inject is None else [(inject, ref_inject)])
    for got, want in pairs:
        assert got.dtype == want.dtype, (got.dtype, want.dtype)
        assert got.shape == want.shape, (got.shape, want.shape)
        assert got.is_contiguous()
        ratio = gate_ratio(got, want)
        assert ratio <= 1.0, f"{ratio:.4g}x the band"


DEV = "cuda"
HC_COUNT = 4
HIDDEN = 2560
WIDTH = HC_COUNT * HIDDEN
LOWRANK = 320
RMS_NORM_EPS = 1e-6

# Element statistics of this checkpoint's layer-0 attn_hyper_connection tensors.
# The normed stream is scaled by (1 + gamma), so gamma's tail -- |6.625| -- is
# what puts it at ~+-30 rather than ~+-4, and the error floor scales with it.
CKPT_DOWN_RMS = 0.0296
CKPT_UP_RMS = 0.0681
CKPT_INJECT_RMS = 0.0107
CKPT_GAMMA_RMS = 0.477
CKPT_GAMMA_ABSMAX = 6.625


def _make_weights(seed, dtype=torch.bfloat16, hc_count=HC_COUNT):
    g = torch.Generator(device=DEV).manual_seed(seed)
    width = hc_count * HIDDEN

    def normal(*shape, scale):
        return (torch.randn(*shape, generator=g, device=DEV) * scale).to(dtype)

    down = normal(LOWRANK, width, scale=CKPT_DOWN_RMS)
    up = normal(width, LOWRANK, scale=CKPT_UP_RMS)
    inject = normal(hc_count, width, scale=CKPT_INJECT_RMS)
    gamma = normal(width, scale=CKPT_GAMMA_RMS)
    # A Gaussian does not reproduce the real gamma's ~14-sigma tail; plant it.
    gamma[0] = CKPT_GAMMA_ABSMAX
    gamma[width // 2] = -CKPT_GAMMA_ABSMAX
    return down, up, inject, gamma


def _grouped_gemma_rmsnorm(x, gamma, hc_count):
    """The hc_norm the entry's input comes from -- here only to give the
    operand its real distribution, not under test."""
    hidden = x.shape[-1] // hc_count
    xf = x.float().reshape(*x.shape[:-1], hc_count, hidden)
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_NORM_EPS)
    return (xf.flatten(-2) * (1.0 + gamma.float())).to(x.dtype)


def _build(leading, seed, hc_count=HC_COUNT, dtype=torch.bfloat16, inject=True):
    down, up, inject_w, gamma = _make_weights(seed, dtype, hc_count)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    raw = torch.randn(*leading, hc_count * HIDDEN, generator=g, device=DEV).to(dtype)
    normed = _grouped_gemma_rmsnorm(raw, gamma, hc_count)
    return raw, (normed, down, up, inject_w if inject else None, hc_count)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    hc_count = spec.get("hc_count", HC_COUNT)
    _, args = _build(
        spec["leading"],
        seed=zlib.crc32(cell.why.encode()),
        hc_count=hc_count,
        dtype=spec.get("dtype", torch.bfloat16),
        inject=spec.get("inject", True),
    )
    before = args[0].clone()
    mixed, inject = op(*args)
    assert mixed.shape == (*spec["leading"], HIDDEN)
    if spec.get("inject", True):
        assert inject.shape == (*spec["leading"], hc_count)
    assert torch.equal(args[0], before), "the normed stream is read-only"
    _compare((mixed, inject), op.reference(*args))


def _wrong_mixed(normed, down, up, hc_count, kind):
    """`mixed` under one mis-reading of the formula, in float64, rounded once."""
    hidden = normed.shape[-1] // hc_count
    x = normed.double()
    logit = x @ down.double().t()
    low = F.silu(logit if kind == "logit_not_divided_by_C" else logit / hc_count)
    gate_logits = low @ up.double().t()
    gate = torch.sigmoid(gate_logits).unflatten(-1, (hc_count, hidden))
    streams = x.unflatten(-1, (hc_count, hidden))
    if kind == "stream_sum":
        mixed = (gate * streams).sum(dim=-2)
    elif kind == "ungated_mean":
        mixed = streams.mean(dim=-2)
    elif kind == "mean_gate_times_mean_stream":
        mixed = gate.mean(dim=-2) * streams.mean(dim=-2)
    elif kind == "hidden_major_gate":
        gate = torch.sigmoid(gate_logits).unflatten(-1, (hidden, hc_count)).transpose(-1, -2)
        mixed = (gate * streams).mean(dim=-2)
    else:
        mixed = (gate * streams).mean(dim=-2)
    return mixed.to(normed.dtype)


@pytest.mark.parametrize(
    "kind",
    [
        "logit_not_divided_by_C",
        "stream_sum",
        "ungated_mean",
        "mean_gate_times_mean_stream",
        "hidden_major_gate",
    ],
)
def test_the_band_refuses_a_wrong_formula(kind) -> None:
    """The band is wider than torch's bf16 `atol`, so it has to be shown to
    refuse what it exists to catch. Measured on sm_103 at the 1024-row MTP
    cell: the op at 0.276 of the band, these five at 31.9x..192x; refused
    here at 10x, so the margin is stated rather than assumed."""
    _, (normed, down, up, _, c) = _build((1024,), seed=1042, inject=False)
    mixed, _ = op(normed, down, up, None, c)
    want, _ = op.reference(normed, down, up, None, c)
    assert gate_ratio(mixed, want) <= 1.0
    assert gate_ratio(_wrong_mixed(normed, down, up, c, kind), want) > 10.0


def test_malformed_operands_raise_rather_than_return() -> None:
    """Why there is no `is_valid`: a width the weights do not take is loud on its own."""
    _, (normed, down, up, inject_w, c) = _build((32,), seed=9)
    with pytest.raises(RuntimeError):
        op(normed[:, : WIDTH - 8], down, up, inject_w, c)


def test_cuda_graph_capture_and_replay() -> None:
    _, (static_in, down, up, inject_w, c) = _build((8,), seed=8)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            op(static_in, down, up, inject_w, c)
    torch.cuda.current_stream().wait_stream(warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_mixed, graph_inject = op(static_in, down, up, inject_w, c)

    _, (fresh, *_) = _build((8,), seed=18)
    static_in.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    eager_mixed, eager_inject = op(fresh, down, up, inject_w, c)
    assert torch.equal(graph_mixed, eager_mixed)
    assert torch.equal(graph_inject, eager_inject)
