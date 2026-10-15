# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4_exp_gated_residual entry.

The thing under test is a formula -- which operand is gated, which axis is
reduced, where each division by `hc_count` sits, the factor on the injection
-- and every mis-reading of it keeps the shapes. So beyond the cells, one test
measures how far outside the entry's band each wrong variant lands.
"""

import zlib
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_gated_residual import (  # noqa: E501
    gate_ratio,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.hc.qwen4_exp_gated_residual import (  # noqa: E501
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


def test_stack_final_mixer_matches_the_injecting_call() -> None:
    """The injection branch is independent of `mixed`, bitwise."""
    _, (normed, down, up, inject_w, c) = _build((256,), seed=1)
    mixed_only, none_weights = op(normed, down, up, None, c)
    assert none_weights is None
    with_inject, _ = op(normed, down, up, inject_w, c)
    assert torch.equal(mixed_only, with_inject)


def test_flattened_call_matches_batched() -> None:
    _, (normed, down, up, inject_w, c) = _build((3, 17), seed=2)
    batched_mixed, batched_inject = op(normed, down, up, inject_w, c)
    flat_mixed, flat_inject = op(normed.reshape(-1, WIDTH), down, up, inject_w, c)
    assert torch.equal(flat_mixed, batched_mixed.reshape(-1, HIDDEN))
    assert torch.equal(flat_inject, batched_inject.reshape(-1, HC_COUNT))


def test_far_above_the_checkpoint_range_the_outputs_stay_bounded() -> None:
    """Finite, `|mixed| <= max|normed|`, `inject` in [0, 2], to 4096x the
    stream. The float64 comparison is not the question up there: the sigmoid
    saturates and bf16 rounding of its logit flips an all-but-binary gate."""
    _, (base, down, up, inject_w, c) = _build((256,), seed=6)
    assert base.float().abs().max().item() > 20.0, "operands are not at the checkpoint's range"
    for scale in (1.0, 8.0, 64.0, 512.0, 4096.0):
        # Scale the normed stream: an RMS norm would undo a scaled input.
        normed = (base.float() * scale).to(torch.bfloat16)
        mixed, inject = op(normed, down, up, inject_w, c)
        assert torch.isfinite(mixed.float()).all(), scale
        assert torch.isfinite(inject.float()).all(), scale
        bound = normed.float().abs().max().item() * (1.0 + torch.finfo(torch.bfloat16).eps)
        assert mixed.float().abs().max().item() <= bound, scale
        assert (inject.float() >= 0.0).all() and (inject.float() <= 2.0).all(), scale


def test_wrong_variants_land_outside_the_band() -> None:
    """Each mis-reading keeps the shapes; each must land > 5x outside the band
    the correct result sits inside."""
    raw, args = _build((512,), seed=7)
    normed, down, up, inject_w, c = args
    want_mixed, want_inject = op.reference(*args)
    got_mixed, got_inject = op(*args)
    print(f"\n  correct mixed  {gate_ratio(got_mixed, want_mixed):.4g}x the band")
    print(f"  correct inject {gate_ratio(got_inject, want_inject):.4g}x the band")

    x = normed.double()
    streams = x.unflatten(-1, (c, HIDDEN))
    logits = x @ down.double().t()
    gate = torch.sigmoid(F.silu(logits / c) @ up.double().t()).unflatten(-1, (c, HIDDEN))
    inject_logits = x @ inject_w.double().t()
    wrong_mixed = {
        "sum instead of mean": lambda: (gate * streams).sum(dim=-2),
        "no /hc_count before silu": lambda: (
            torch.sigmoid(F.silu(logits) @ up.double().t()).unflatten(-1, (c, HIDDEN)) * streams
        ).mean(dim=-2),
        "silu and sigmoid swapped": lambda: (
            F.silu(torch.sigmoid(logits / c) @ up.double().t()).unflatten(-1, (c, HIDDEN)) * streams
        ).mean(dim=-2),
        "stream axis taken last": lambda: (
            gate.transpose(-1, -2).reshape(-1, c, HIDDEN) * streams
        ).mean(dim=-2),
        "gates the raw stream": lambda: (gate * raw.double().unflatten(-1, (c, HIDDEN))).mean(
            dim=-2
        ),
        "mean over hidden": lambda: (gate * streams)
        .mean(dim=-1)
        .repeat_interleave(HIDDEN // c, dim=-1),
    }
    wrong_inject = {
        "missing the factor of 2": lambda: torch.sigmoid(inject_logits / c),
        "no /hc_count on the logit": lambda: 2.0 * torch.sigmoid(inject_logits),
        "tanh instead of 2*sigmoid": lambda: torch.tanh(inject_logits / c),
        "silu instead of 2*sigmoid": lambda: F.silu(inject_logits / c),
    }
    for want, variants in ((want_mixed, wrong_mixed), (want_inject, wrong_inject)):
        for name, build in variants.items():
            ratio = gate_ratio(build().to(torch.bfloat16), want)
            print(f"  {name:<28} {ratio:10.4g}x the band")
            assert ratio > 5.0, f"the band does not separate {name!r} ({ratio:.4g}x)"
            with pytest.raises(AssertionError):
                _compare((build().to(torch.bfloat16), None), (want, None))


def test_malformed_operands_raise_rather_than_return() -> None:
    """Why there is no `is_valid`: each of these is loud on its own."""
    _, (normed, down, up, inject_w, c) = _build((32,), seed=9)
    with pytest.raises(RuntimeError):
        op(normed[:, : WIDTH - 8], down, up, inject_w, c)
    with pytest.raises(RuntimeError):
        op(normed, down.float(), up, inject_w, c)
    odd = WIDTH + 1
    with pytest.raises(RuntimeError):
        op(
            torch.randn(32, odd, dtype=torch.bfloat16, device=DEV),
            torch.randn(LOWRANK, odd, dtype=torch.bfloat16, device=DEV),
            torch.randn(odd, LOWRANK, dtype=torch.bfloat16, device=DEV),
            torch.randn(3, odd, dtype=torch.bfloat16, device=DEV),
            3,
        )


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
