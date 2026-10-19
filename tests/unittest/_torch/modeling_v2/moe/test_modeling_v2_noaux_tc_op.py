# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the noaux_tc_op catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` is the
mirror, `is_valid` is the guard. This file owns what a cell cannot carry:
- turning a spec into tensors (two cells read real router biases from the
  nemotron-3 ultra checkpoint);
- the band the weights are held to;
- HF's own selection as a second witness;
- the tie rule and the fp32 bias add, each driven where it decides the answer;
- the inputs the guard exists to refuse.
"""

import json
import math
import os
import zlib
from functools import lru_cache

import pytest
import torch
from safetensors import safe_open
from utils.llm_data import get_checkpoint

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import noaux_tc_op as op

__extra_import_path__ = [".."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    assert_within_ulp,
    certifying,
    validating,
)

assert torch.cuda.is_available(), "noaux_tc_op requires a CUDA device"

#: The checkpoint the ("checkpoint", layer) cells read their bias from, under
#: LLM_MODELS_ROOT. Missing, it fails loudly rather than skipping a cell.
ULTRA_CHECKPOINT = "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"

#: How far the kernel's weights may sit from `reference`'s, per output dtype, in
#: `_validating.ULP` units against each row's largest weight.
#:
#: Both sides make the same selection from the same scores. They differ only in
#: the normalization:
#: - the kernel sums the selected scores with a 5-level fp32 warp tree, then
#:   divides in fp64 and rounds once. Every term is positive, so the tree's
#:   relative error is at most 5u, with u = 2^-24;
#: - `reference` does all of it in fp64 and rounds once.
#:
#: fp32 out: the tree's 5u plus one rounding on each side gives 7u of each
#: weight. That is 3.5 ULP (ULP = 2u) of the row's largest. Measured: <= 1.5
#: per element, <= 0.45 RMS.
#: bf16 out: values 6u apart round to the same bf16 or to neighbours, and one
#: bf16 ulp of a weight is at most 2^-7 of it. That is 2 ULP (ULP = 2^-8) of the
#: row's largest. Measured: bit-exact.
WEIGHT_BAND = {
    torch.float32: dict(element_ulp=3.5, rms_ulp=3.5),
    torch.bfloat16: dict(element_ulp=2.0, rms_ulp=2.0),
}

#: The single-group cells: nemotron-3 ultra's router.
ULTRA_CELLS = [c for c in op.CELLS if c.spec["n_group"] == 1]


@lru_cache(maxsize=None)
def _checkpoint_bias(layer: int) -> torch.Tensor:
    """One MoE layer's `gate.e_score_correction_bias`, read alone from its shard."""
    root = get_checkpoint(ULTRA_CHECKPOINT)
    key = f"backbone.layers.{layer}.mixer.gate.e_score_correction_bias"
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        shard = json.load(f)["weight_map"][key]
    with safe_open(os.path.join(root, shard), framework="pt") as f:
        return f.get_tensor(key)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    tokens, experts = spec["tokens"], spec["experts"]
    logits = torch.randn(tokens, experts, generator=g, device="cuda", dtype=torch.float32)
    logits = (logits * spec["logit_std"]).to(spec["logits_dtype"])
    kind, *params = spec["bias"]
    if kind == "normal":
        mean, std = params
        bias = torch.randn(experts, generator=g, device="cuda", dtype=torch.float32) * std + mean
    elif kind == "uniform":
        lo, hi = params
        bias = torch.rand(experts, generator=g, device="cuda", dtype=torch.float32) * (hi - lo) + lo
    else:
        assert kind == "checkpoint", f"unknown bias kind {kind!r}"
        bias = _checkpoint_bias(*params).to("cuda")
        assert bias.dtype is torch.float32 and bias.shape == (experts,), (bias.dtype, bias.shape)
    return logits, bias, spec["n_group"], spec["topk_group"], spec["topk"], spec["scale"]


def _seed(cell):
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    args = _build(cell.spec, _seed(cell))
    weights, ids = op(*args)
    ref_weights, ref_ids = op.reference(*args)

    tokens, topk = cell.spec["tokens"], cell.spec["topk"]
    assert ids.dtype is torch.int32 and ids.shape == (tokens, topk) and ids.is_contiguous()
    assert weights.dtype is cell.spec["logits_dtype"], "the weights take the logits' dtype"
    assert weights.shape == (tokens, topk) and weights.is_contiguous()
    # Ids are an exact claim, order included: routing either picks the same
    # experts or it does not, and a near-miss here is a different set of
    # experts, not a rounding. Exact ties do not relax it -- both sides send a
    # tie to the lower id, and the layer-1 cell carries hundreds of them at
    # the top-k boundary.
    assert torch.equal(ids, ref_ids), "selected experts differ from the reference"
    assert_within_ulp(weights, ref_weights, **WEIGHT_BAND[weights.dtype])


def test_a_tie_goes_to_the_lower_expert_id(certifying) -> None:  # noqa: F811
    """Drive the tie rule where it decides the answer.

    Experts 7 and 300 carry bias 38.25 and every other expert 38.0, so equal
    fp32 biased scores can come from unequal sigmoids.
    - Rows 0 and 1: 21 experts (100-120) win clearly, then two experts tie
      exactly for 22nd place at 38.5. In row 0 that is 5 (sigmoid 0.5) against
      300 (0.25). Row 1 mirrors it: 7 (0.25) against 400 (0.5). The lower id
      must win both, whichever sigmoid it holds.
    - Row 2: every logit is 0. Experts 7 and 300 lead, and a 510-way tie fills
      the rest in ascending id order.
    In every row, a winner's weight must come from its own unbiased sigmoid.
    """
    certifying(op)
    experts, topk, scale = 512, 22, 5.0
    logits = torch.full((3, experts), -4.0, device="cuda")
    bias = torch.full((experts,), 38.0, device="cuda")
    bias[[7, 300]] = 38.25
    winners = list(range(100, 121))
    logits[:2, winners] = 4.0
    half, quarter = 0.0, math.log(1 / 3)  # sigmoid 0.5 and 0.25
    logits[0, 5], logits[0, 300] = half, quarter
    logits[1, 7], logits[1, 400] = quarter, half
    logits[2] = 0.0

    weights, ids = op(logits, bias, 1, 1, topk, scale)
    ref_weights, ref_ids = op.reference(logits, bias, 1, 1, topk, scale)

    assert ids[0].tolist() == winners + [5]
    assert ids[1].tolist() == winners + [7]
    assert ids[2].tolist() == [7, 300] + [e for e in range(21) if e != 7]
    assert torch.equal(ids, ref_ids)
    assert_within_ulp(weights, ref_weights, **WEIGHT_BAND[torch.float32])
    # The bias picked them; their unbiased sigmoids weigh them.
    win = 1 / (1 + math.exp(-4.0))
    torch.testing.assert_close(
        weights[:2, -1] / weights[:2, 0],
        torch.tensor([0.5 / win, 0.25 / win], device="cuda"),
        rtol=2**-20,
        atol=0.0,
    )
    torch.testing.assert_close(
        weights[2], torch.full((topk,), scale / topk, device="cuda"), rtol=2**-21, atol=0.0
    )


@pytest.mark.parametrize("cell", ULTRA_CELLS, ids=[c.why[:44] for c in ULTRA_CELLS])
def test_picks_what_the_hf_router_picks_up_to_ties(cell) -> None:
    """HF's selection, in HF's own fp32, as a second witness for the ultra cells.

    `reference` makes the kernel's selection bit for bit, in the kernel's
    arithmetic. HF's own code makes it differently in two ways:
    - it computes the sigmoid as 1/(1+exp(-x)), a few 2^-24 from the tanh form
      (measured <= 2);
    - it picks with `torch.topk`, whose tie order is unspecified.
    After the fp32 bias add, the sigmoid difference can move a biased score by
    one fp32 step q: 2^-19 at 25, 2^-18 at 38. The 22nd-place score can move
    by q as well.

    So the kernel must pick what HF picks, except among experts within 2q of
    HF's 22nd-place score. Anything further above must be picked, and anything
    further below must not. Measured: the picks differ only in the layer-1
    cell, on 7 of its 8192 tokens, and only by experts within one step.

    The weights need no second witness. `reference` evaluates HF's formula
    (unbiased scores / (sum + 1e-20) * scale), and the tie test checks that a
    winner is weighted by its unbiased sigmoid.
    """
    logits, bias, n_group, topk_group, topk, scale = _build(cell.spec, _seed(cell))
    with validating(op):
        _, ids = op(logits, bias, n_group, topk_group, topk, scale)

    choice = torch.sigmoid(logits.float()) + bias.float()
    kth = torch.topk(choice, topk, dim=-1).values[:, -1:]
    q = torch.nextafter(kth, torch.full_like(kth, math.inf)) - kth
    picked = torch.zeros_like(choice, dtype=torch.bool).scatter_(1, ids.long(), True)
    assert picked.sum(-1).eq(topk).all(), "an expert was picked twice in one row"
    assert not (picked & (choice < kth - 2 * q)).any(), "picked an expert below the boundary"
    assert not (~picked & (choice > kth + 2 * q)).any(), "skipped an expert above the boundary"


def test_the_bias_add_is_fp32() -> None:
    """The cells can tell an fp32 bias add from a bf16 one, and the kernel's is fp32.

    The source widens the bias and adds in fp32. This holds the kernel to that
    observably, on layer 107's checkpoint bias. A bf16 add would round each
    biased score to a step of 1/4, collapsing the sigmoid's whole range into
    five levels, and would pick different experts on nearly every token
    (measured: all 32; the gate is 90%).
    """
    cell = next(c for c in ULTRA_CELLS if c.spec["bias"] == ("checkpoint", 107))
    logits, bias, n_group, topk_group, topk, scale = _build(cell.spec, _seed(cell))
    with validating(op):
        _, ids = op(logits, bias, n_group, topk_group, topk, scale)

    def pick(choice):
        return torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :topk].int()

    scores = 0.5 * torch.tanh(0.5 * logits) + 0.5
    assert torch.equal(ids, pick(scores + bias)), "the kernel's picks are not the fp32 add's"
    bf16_add = pick((scores.bfloat16() + bias.bfloat16()).float())
    assert (bf16_add != ids).any(-1).float().mean() > 0.9, "a bf16 add no longer looks different"


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


def _ok():
    logits = torch.randn(4, 16, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(16, dtype=torch.float32, device="cuda")
    return logits, bias


def test_guard_refuses_strided_logits() -> None:
    """Read as a dense buffer from data_ptr(); strides are ignored."""
    logits = torch.randn(4, 32, dtype=torch.bfloat16, device="cuda")[:, ::2]
    _, bias = _ok()
    with pytest.raises(AssertionError, match="router_logits must be contiguous"):
        with validating(op):
            op(logits, bias, 1, 1, 2, 1.0)


def test_guard_refuses_a_strided_bias() -> None:
    logits, _ = _ok()
    bias = torch.randn(32, dtype=torch.float32, device="cuda")[::2]
    with pytest.raises(AssertionError, match="bias must be contiguous"):
        with validating(op):
            op(logits, bias, 1, 1, 2, 1.0)


def test_guard_refuses_topk_past_the_expert_count() -> None:
    """Reads out of bounds and emits ids outside the expert range."""
    logits, bias = _ok()
    with pytest.raises(AssertionError, match="exceeds num_experts"):
        with validating(op):
            op(logits, bias, 1, 1, 17, 1.0)
