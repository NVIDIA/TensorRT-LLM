# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the noaux_tc_op catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import (  # noqa: E501
    noaux_tc_op as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "noaux_tc_op requires a CUDA device"

# Step-5-Preview's router: 352 experts, one group, top-8, weights scaled by 3.0.
_STEP5 = dict(n_group=1, topk_group=1, topk=8, routed_scaling_factor=3.0)
_STEP5_EXPERTS = 352


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        spec["tokens"], spec["experts"], generator=g, device="cuda", dtype=torch.float32
    ).to(spec.get("logits_dtype", torch.bfloat16))
    bias = torch.randn(spec["experts"], generator=g, device="cuda", dtype=torch.float32)
    return logits, bias, spec["n_group"], spec["topk_group"], spec["topk"], spec["scale"]


def _compare(weights, ids, ref_weights, ref_ids) -> None:
    """Ids exactly; fp32 weights bit for bit, 16-bit weights in torch's band.

    Ids are an exact claim: routing either picks the same experts or it does
    not, and a near-miss is a different set of experts, not a rounding.

    fp32 weights (every Step-5 input: the router hands over fp32 logits) are
    held bit-exact, and the band is zero by derivation: `reference` follows
    the kernel's arithmetic step for step -- the tanh-form sigmoid, the fp32
    bias add, an fp32 sum of the selected sigmoids, and a normalization
    evaluated in fp64 and rounded once to fp32. The one freedom left is the
    order of the fp32 sum of eight values (the kernel's warp reduction
    against torch's), and on every fp32 input here the two orders land on
    the same total. A wrong sigmoid, the bias leaking into the weights or a
    missing scale moves a weight by far more than an ulp.

    bf16 weights (the deepseek-r1 cells) are held in torch's dtype default:
    the kernel's own fp64 -> bf16 conversion is not torch's, and on the
    deepseek 8192-token cell the two disagree by one bf16 ulp (max |diff|
    1.953e-03 at weights near 0.4).
    """
    assert torch.equal(ids, ref_ids), "selected experts differ from the reference"
    assert ids.dtype is torch.int32
    assert weights.dtype is ref_weights.dtype
    if weights.dtype is torch.float32:
        assert torch.equal(weights, ref_weights), (
            "fp32 weights differ from the reference: max |diff| "
            f"{(weights.double() - ref_weights.double()).abs().max().item():.3e}"
        )
    else:
        torch.testing.assert_close(weights, ref_weights)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    weights, ids = op(*args)
    ref_weights, ref_ids = op.reference(*args)
    _compare(weights, ids, ref_weights, ref_ids)
    if cell.spec["experts"] == _STEP5_EXPERTS:
        # The 512-expert instantiation pads 352..511 with -inf keys; an id in
        # the padding would mean the pad was read as an expert.
        assert int(ids.max()) < _STEP5_EXPERTS


def test_the_sigmoid_is_the_tanh_form() -> None:
    """The entry's reference claims a specific sigmoid; hold the op to it.

    `1/(1+exp(-x))` and `0.5*tanh(x/2)+0.5` agree everywhere a random logit
    lands and diverge in the tails, where the tanh form saturates exactly. A
    cell over random inputs cannot tell them apart, so this drives the tails
    directly -- otherwise `reference` could encode the wrong sigmoid and every
    cell would still pass.
    """
    experts = 8
    logits = torch.full((4, experts), -30.0, dtype=torch.bfloat16, device="cuda")
    logits[:, 0] = 30.0
    bias = torch.zeros(experts, dtype=torch.float32, device="cuda")
    weights, _ = op(logits, bias, 1, 1, 2, 1.0)
    ref_weights, _ = op.reference(logits, bias, 1, 1, 2, 1.0)
    torch.testing.assert_close(weights, ref_weights)


@pytest.mark.parametrize("tokens", [1, 256, 257, 8192])
def test_step5_ties_break_toward_the_lower_expert_id(tokens) -> None:
    """Tied selection scores at 352 experts, in both ungrouped top-k paths.

    The kernel ranks experts by a packed (score, 65535 - id) key, and its
    small-batch path takes the top 8 of each 128-expert batch before ranking
    the candidates. A tie split across batch boundaries (127 | 128, 255 |
    256) or reaching into the last, partial batch (256..351, padded to 511)
    is where an ordering defect in either level would show. 256 and 257
    tokens sit either side of the default small/large-batch crossover, so
    both implementations run on the same ties.
    """
    logits = torch.full((tokens, _STEP5_EXPERTS), -4.0, dtype=torch.float32, device="cuda")
    bias = torch.zeros(_STEP5_EXPERTS, dtype=torch.float32, device="cuda")
    tied = [351, 300, 256, 255, 128, 127, 64, 3, 2]  # a nine-way tie for seven slots
    logits[:, tied] = 1.0
    logits[:, 200] = 2.0  # one clear winner
    logits[:, 0] = 0.99  # the lowest id, just below the tie
    weights, ids = op(logits, bias, **_STEP5)
    ref_weights, ref_ids = op.reference(logits, bias, **_STEP5)
    _compare(weights, ids, ref_weights, ref_ids)
    assert ids[0].tolist() == [200, 2, 3, 64, 127, 128, 255, 256]


def test_step5_cuda_graph_capture_and_replay() -> None:
    """The decode target captures the router; replay must re-route new logits.

    Captured at 128 rows (the small-batch path, tp4/tep4 decode) and 512 rows
    (the large-batch path, dep4 decode after the all-gather). The outputs
    are allocated inside the capture, so every replay writes the same
    tensors; each replay is checked against `reference` on logits refilled
    after the capture.
    """
    for tokens in (128, 512):
        spec = dict(
            tokens=tokens,
            experts=_STEP5_EXPERTS,
            n_group=1,
            topk_group=1,
            topk=8,
            scale=3.0,
            logits_dtype=torch.float32,
        )
        logits, bias, *rest = _build(spec, seed=tokens)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            op(logits, bias, *rest)  # warm-up outside the capture
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            weights, ids = op(logits, bias, *rest)
        for seed in (tokens + 1, tokens + 2):
            fresh, *_ = _build(spec, seed=seed)
            logits.copy_(fresh)
            graph.replay()
            torch.cuda.synchronize()
            _compare(weights, ids, *op.reference(logits, bias, *rest))


def _ok():
    torch.manual_seed(0)
    logits = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(16, dtype=torch.float32, device="cuda")
    return logits, bias


def test_guard_refuses_strided_logits() -> None:
    """Read as a dense buffer from data_ptr(); strides are ignored."""
    logits = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")[:, ::2]
    _, bias = _ok()
    # Unguarded, the op answers without complaint -- and routes wrongly.
    _, ids = op(logits, bias, 1, 1, 2, 1.0)
    _, dense_ids = op(logits.contiguous(), bias, 1, 1, 2, 1.0)
    assert not torch.equal(ids, dense_ids)
    with pytest.raises(AssertionError, match="router_logits must be contiguous"):
        with validating(op):
            op(logits, bias, 1, 1, 2, 1.0)


def test_guard_refuses_a_strided_bias() -> None:
    logits, _ = _ok()
    bias = torch.randn(32, dtype=torch.float32, device="cuda")[::2]
    # Unguarded, the op answers without complaint -- and routes wrongly.
    _, ids = op(logits, bias, 1, 1, 2, 1.0)
    _, dense_ids = op(logits, bias.contiguous(), 1, 1, 2, 1.0)
    assert not torch.equal(ids, dense_ids)
    with pytest.raises(AssertionError, match="bias must be contiguous"):
        with validating(op):
            op(logits, bias, 1, 1, 2, 1.0)


def test_guard_refuses_topk_past_the_expert_count() -> None:
    """Reads out of bounds and emits ids outside the expert range."""
    logits, bias = _ok()
    with pytest.raises(AssertionError, match="exceeds num_experts"):
        with validating(op):
            op(logits, bias, 1, 1, 17, 1.0)
