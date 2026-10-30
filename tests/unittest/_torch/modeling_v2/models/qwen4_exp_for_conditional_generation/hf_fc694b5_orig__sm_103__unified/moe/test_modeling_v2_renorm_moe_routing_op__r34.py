# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen4 unified target's renorm_moe_routing_op variant
(ops/moe/renorm_moe_routing_op__r34), copied from the tp1 target's test.

The entry owns the contract. This file turns a cell into logits and holds the
claims no cell carries: equivalence to HF's `norm_topk_prob` routing, and the
contiguity guard.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified.ops.moe.renorm_moe_routing_op__r34 import (  # noqa: E501
    renorm_moe_routing_op as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "renorm_moe_routing_op requires a CUDA device"


def _compare(
    out: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Ids equal and weights bit-exact.

    The reference is not an approximation of the kernel: over 2040
    configurations and 16384-token runs at 512 / top-10, not one id and not
    one weight bit differed. A one-ulp `assert_close` runs first only so a
    drift reports a readable diff. The nearest wrong answers sit 0.4476
    away (un-renormalized softmax, `1/topk`), and a `torch.topk` reference
    differs on a third of the rows at 512 bf16 experts.
    """
    ids, weights = out
    ref_ids, ref_weights = ref
    assert ids.dtype == ref_ids.dtype == torch.int32, (ids.dtype, ref_ids.dtype)
    assert weights.dtype == ref_weights.dtype, (weights.dtype, ref_weights.dtype)
    assert ids.shape == ref_ids.shape == weights.shape, (ids.shape, ref_ids.shape)
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
    eps = torch.finfo(weights.dtype).eps
    torch.testing.assert_close(weights, ref_weights, rtol=eps, atol=0.0)
    mismatched = int((weights != ref_weights).sum())
    assert mismatched == 0, f"{mismatched} of {weights.numel()} weights are not bit-exact"


DEV = "cuda"


def _logits(tokens, num_experts, logits_dtype, dist="randn", magnitude=1.0, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    if dist == "coarse":
        x = torch.randint(0, 3, (tokens, num_experts), device=DEV, generator=g).float()
    else:
        x = torch.randn(tokens, num_experts, device=DEV, generator=g) * magnitude
    return x.to(logits_dtype)


def _check(logits, topk, output_dtype=None):
    """Drive the op and gate it through the entry's own reference and `_compare`."""
    ids, weights = op(logits, topk, output_dtype)
    assert ids.is_contiguous() and weights.is_contiguous()
    assert ids.device == weights.device == logits.device
    _compare((ids, weights), op.reference(logits, topk, output_dtype))
    return ids, weights


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    s = cell.spec
    logits = _logits(
        s["tokens"],
        s["num_experts"],
        s["logits_dtype"],
        s["dist"],
        s.get("magnitude", 1.0),
        seed=zlib.crc32(cell.why.encode()),
    )
    ids, weights = _check(logits, s["topk"], s["output_dtype"])
    assert weights.dtype == s["output_dtype"]
    # renormalized: every row sums to 1
    torch.testing.assert_close(
        weights.float().sum(-1),
        torch.ones(s["tokens"], device=DEV),
        rtol=8e-3 if s["output_dtype"] is torch.bfloat16 else 1e-6,
        atol=0.0,
    )


# ── what a cell cannot carry ──────────────────────────────────────────────


def test_equivalent_to_hf_norm_topk_prob():
    """Softmax over all experts, top-k, divide by the selected sum: the same
    operator up to rounding (5.96e-08 measured); skipping the renormalization
    lands three orders of magnitude outside that."""
    logits = _logits(1024, 512, torch.bfloat16, magnitude=2.0, seed=2)
    _, weights = op(logits, 10, torch.float32)
    hf_values, _ = torch.topk(torch.softmax(logits.float(), dim=-1), 10, dim=-1)
    hf_weights = hf_values / hf_values.sum(-1, keepdim=True)
    # ids can differ on tie rows; the sorted weight multiset cannot
    torch.testing.assert_close(
        weights.sort(-1, descending=True).values,
        hf_weights.sort(-1, descending=True).values,
        rtol=1e-5,
        atol=1e-7,
    )
    assert (weights - hf_values).abs().max().item() > 1e-2


def test_cuda_graph_replay_at_the_captured_batch():
    """The trunk's decode router runs inside the engine's CUDA graphs: captured
    once at the largest captured batch (128 rows), each replay over refilled
    logits answers bit-exactly what the reference does on them."""
    tokens, num_experts, topk = 128, 512, 10
    logits = _logits(tokens, num_experts, torch.bfloat16, seed=21)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(logits, topk, torch.bfloat16)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, weights = op(logits, topk, torch.bfloat16)
    for seed in (22, 23):
        logits.copy_(_logits(tokens, num_experts, torch.bfloat16, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        _compare(
            (ids.clone(), weights.clone()),
            op.reference(logits, topk, torch.bfloat16),
        )


# ── the guard ─────────────────────────────────────────────────────────────


def test_guard_refuses_a_column_slice():
    """What `is_valid` exists for: the raw op reads a slice as the dense buffer."""
    wide = _logits(8, 1024, torch.bfloat16, seed=12)
    with validating(op), pytest.raises(AssertionError, match="contiguous"):
        op(wide[:, :512], 10)
