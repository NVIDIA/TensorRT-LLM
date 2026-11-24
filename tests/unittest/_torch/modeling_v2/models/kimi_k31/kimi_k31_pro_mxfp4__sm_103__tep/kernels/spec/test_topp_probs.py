# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cluster top-p kernel's paths, bit for bit: the candidate push against the cluster-wide fallback
passes (``force_slow``), the one-warp last rank against the last 10-bit histogram (``warp_last``),
PDL on against off; the tie rule; the flat rows that take the fallback by themselves; and a
16-CTA cluster against the reference."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.kernels.spec.topp_probs import (  # noqa: E501
    CAND_TOTAL,
    DEFAULT_CLUSTER,
    topp_probs,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (  # noqa: E501
    filtered_probs,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    get_sm_version() != 103, reason="the K3.1 cluster top-p kernel is certified on sm_103"
)

VOCAB = 163840
LOG2E = 1.4426950408889634


def _make(kind: str, rows: int, seed: int):
    """``(logits, temperatures, top_p)``: float32 logits of ``kind``, T = 1, top-p 0.95."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, VOCAB, generator=g, device="cuda") * 3.0
    if kind == "peaked":
        hot = torch.rand(rows, VOCAB, generator=g, device="cuda").topk(3, dim=-1).indices
        x = x.scatter(
            1, hot, x.amax(-1, keepdim=True) + torch.tensor([6.0, 5.0, 4.0], device="cuda")
        )
    elif kind == "ties":
        # BF16 values: thousands of exact ties around any cut.
        x = x.bfloat16().float()
    elif kind == "lmlike":
        # Most of the vocabulary far below a few hundred plausible tokens.
        x = torch.randn(rows, VOCAB, generator=g, device="cuda") * 2.0 - 8.0
        top = torch.rand(rows, VOCAB, generator=g, device="cuda").topk(256, dim=-1).indices
        x = x.scatter(1, top, torch.randn(rows, 256, generator=g, device="cuda") * 2.0 + 6.0)
    elif kind == "flat":
        x = torch.randn(rows, VOCAB, generator=g, device="cuda") * 0.5
    else:
        assert kind == "randn", kind
    temperatures = torch.ones(rows, device="cuda")
    top_p = torch.full((rows,), 0.95, device="cuda")
    return x.contiguous(), temperatures, top_p


def _run(x, temperatures, top_p, enable_pdl=False, **hooks):
    return topp_probs(x, temperatures, top_p, torch.empty_like(x), enable_pdl, **hooks)


@pytest.mark.parametrize("rows", [5, 40])
@pytest.mark.parametrize("kind", ["randn", "peaked", "ties", "lmlike", "flat"])
def test_paths_are_bitwise_equal(kind, rows) -> None:
    x, temperatures, top_p = _make(kind, rows, seed=rows)
    base = _run(x, temperatures, top_p)
    assert torch.equal(_run(x, temperatures, top_p), base), "not repeatable"
    for hooks in (
        dict(force_slow=True),
        dict(warp_last=False),
        dict(enable_pdl=True),
        dict(enable_pdl=True, force_slow=True, warp_last=False),
    ):
        assert torch.equal(_run(x, temperatures, top_p, **hooks), base), hooks


def test_tie_rule() -> None:
    """BF16 logits: within a CTA's slice every token at or above the slice's smallest kept logit is
    kept (equal logits get equal weights); across slices the weights of equal logits may differ in
    the last bit, so globally no dropped logit exceeds a kept one beyond rounding."""
    x, temperatures, top_p = _make("ties", 8, seed=3)
    kept = _run(x, temperatures, top_p) > 0
    slice_ = VOCAB // DEFAULT_CLUSTER
    for r in range(x.shape[0]):
        lowest = x[r][kept[r]].min()
        assert x[r][~kept[r]].max() <= lowest + lowest.abs() * 1e-6 + 1e-6, r
        for c in range(DEFAULT_CLUSTER):
            xs, ks = x[r, c * slice_ : (c + 1) * slice_], kept[r, c * slice_ : (c + 1) * slice_]
            if ks.any():
                assert torch.equal(xs >= xs[ks].min(), ks), (r, c)


def test_flat_rows_take_the_fallback() -> None:
    """The flat rows hold more candidates (keys in the crossing 1/8-binade bucket) in some CTA's
    slice than a CTA may push, so the cluster takes the fallback passes by itself; the result
    is still the reference's distribution."""
    x, temperatures, top_p = _make("flat", 4, seed=11)
    probs = _run(x, temperatures, top_p)
    y = x * LOG2E
    weights = torch.exp2(y - y.amax(-1, keepdim=True).ceil())
    bucket = weights.view(torch.int32) >> 20
    slice_ = VOCAB // DEFAULT_CLUSTER
    for r in range(x.shape[0]):
        kept = probs[r] > 0
        d1 = bucket[r][kept][weights[r][kept].argmin()]
        most = (bucket[r] == d1).view(DEFAULT_CLUSTER, slice_).sum(-1).max().item()
        assert most > 2 * (CAND_TOTAL // DEFAULT_CLUSTER), (
            f"row {r}: {most} candidates in one slice"
        )
    expected = filtered_probs(x, temperatures, None, top_p, None)
    assert (probs - expected).abs().sum(-1).max().item() <= 5e-5


@pytest.mark.parametrize("kind", ["randn", "peaked"])
def test_cluster_16(kind) -> None:
    """A 16-CTA cluster (other slices, so not bit-equal to 8) gives the reference's distribution."""
    x, temperatures, top_p = _make(kind, 5, seed=7)
    probs = _run(x, temperatures, top_p, cluster=16)
    assert torch.equal(_run(x, temperatures, top_p, cluster=16, force_slow=True), probs)
    expected = filtered_probs(x, temperatures, None, top_p, None)
    assert (probs - expected).abs().sum(-1).max().item() <= 5e-5
    torch.testing.assert_close(probs.sum(-1), torch.ones(5, device="cuda"), atol=1e-5, rtol=0)
