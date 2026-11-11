# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the noaux_tc_op catalog entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.noaux_tc_op import (  # noqa: E501
    noaux_tc_op as op,
)

__extra_import_path__ = [".."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

assert torch.cuda.is_available(), "noaux_tc_op requires a CUDA device"


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(
        spec["tokens"], spec["experts"], generator=g, device="cuda", dtype=torch.float32
    ).to(spec.get("logits_dtype", torch.bfloat16))
    bias = torch.randn(spec["experts"], generator=g, device="cuda", dtype=torch.float32)
    return logits, bias, spec["n_group"], spec["topk_group"], spec["topk"], spec["scale"]


# The shapes each shipped target calls this entry with, by target directory
# under `models/`. The entry's own `CELLS` are the op's whoever calls it.
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "deepseek_v3/r1_0528_nvfp4__sm_10x__default": (
        Cell(
            why="the default target's expert chunk, which its attention-DP gather fills",
            spec=dict(tokens=8192, experts=256, n_group=8, topk_group=4, topk=8, scale=2.5),
        ),
        Cell(
            why="deepseek MTP router: fp32 logits, so the weights are the fp32 "
            "token_final_scales fused_moe needs",
            spec=dict(
                tokens=256,
                experts=256,
                n_group=8,
                topk_group=4,
                topk=8,
                scale=2.5,
                logits_dtype=torch.float32,
            ),
        ),
    ),
}

CELLS = certified_cells(op, TARGET_CELLS)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    with validating(op):
        weights, ids = op(*args)
    ref_weights, ref_ids = op.reference(*args)

    # Ids are an exact claim: routing either picks the same experts or it does
    # not, and a near-miss here is a different set of experts, not a rounding.
    assert torch.equal(ids, ref_ids), "selected experts differ from the reference"
    assert ids.dtype is torch.int32
    torch.testing.assert_close(weights, ref_weights)


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


@pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="the Kimi K3 cell is certified on sm_100"
)
def test_kimi_k3_config() -> None:
    torch.manual_seed(12)
    bias = torch.randn(896, dtype=torch.float32, device="cuda") * 0.1
    for num_tokens in [1, 8, 64, 65, 512, 2048, 8192]:
        logits = torch.randn(num_tokens, 896, dtype=torch.float32, device="cuda") * 2.5
        weights, ids = op(logits, bias, 1, 1, 16, 1.0)
        ref_weights, ref_ids = op.reference(logits, bias, 1, 1, 16, 1.0)
        assert torch.equal(ids, ref_ids)
        torch.testing.assert_close(
            weights, ref_weights, rtol=torch.finfo(logits.dtype).eps, atol=0.0
        )
        assert int((weights != ref_weights).sum()) <= 1 + weights.numel() // 10000
        assert ids.dtype == torch.int32 and ids.device == logits.device
        assert weights.is_contiguous() and ids.is_contiguous()
        assert int(ids.min()) >= 0 and int(ids.max()) < 896
        assert (torch.sort(ids, dim=-1).values.diff(dim=-1) > 0).all()
        torch.testing.assert_close(
            weights.sum(-1),
            torch.ones(num_tokens, device="cuda"),
            rtol=16 * torch.finfo(torch.float32).eps,
            atol=0.0,
        )
