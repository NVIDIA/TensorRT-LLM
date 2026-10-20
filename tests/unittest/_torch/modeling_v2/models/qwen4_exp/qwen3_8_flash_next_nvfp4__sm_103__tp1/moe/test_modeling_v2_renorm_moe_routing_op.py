# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the renorm_moe_routing_op catalog entry.

The entry owns the contract. This file turns a cell into logits and holds the
claims no cell carries: the return order, the tie-break against `torch.topk`,
the sibling op one flag away, the sweep of the old certified surface, the edge
behaviour, and the guards.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.renorm_moe_routing_op import (  # noqa: E501
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


def test_old_certified_surface():
    """The rest of the surface the entry was certified over, through the entry's
    own reference and gate: the token column in both output dtypes, every input
    dtype against every output dtype, and the num_experts / topk sweep across
    the kernel's power-of-two instantiation boundaries."""
    for tokens in (1, 2, 7, 8, 64, 512, 2048, 4096, 8192):
        logits = _logits(tokens, 512, torch.bfloat16, seed=tokens)
        for out_dtype in (torch.float32, torch.bfloat16):
            _check(logits, 10, out_dtype)
    base = _logits(256, 512, torch.float32, magnitude=2.0, seed=8)
    for in_dtype in (torch.bfloat16, torch.float16, torch.float32):
        for out_dtype in (torch.float32, torch.bfloat16, None):
            _check(base.to(in_dtype), 10, out_dtype)
    for num_experts in (1, 2, 8, 31, 32, 33, 64, 96, 128, 129, 256, 384, 511, 512):
        _check(_logits(64, num_experts, torch.bfloat16, seed=num_experts), min(10, num_experts))
    logits = _logits(256, 512, torch.bfloat16, seed=9)
    for topk in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16):  # topk 0: test_degenerate_shapes
        _check(logits, topk, torch.bfloat16)


# ── the traps ─────────────────────────────────────────────────────────────


def test_return_order_is_ids_then_weights():
    logits = _logits(256, 512, torch.bfloat16, seed=1)
    first, second = op(logits, 10, torch.bfloat16)
    assert first.dtype == torch.int32 and second.dtype == torch.bfloat16
    top1 = torch.sort(logits, dim=-1, descending=True, stable=True).indices[:, 0]
    assert torch.equal(first[:, 0], top1.to(torch.int32))


def test_equivalent_to_hf_norm_topk_prob():
    """Softmax over all experts, top-k, divide by the selected sum: the same
    operator up to rounding (5.96e-08 measured); skipping the renormalization
    lands three orders of magnitude outside that."""
    for dtype in (torch.bfloat16, torch.float32):
        logits = _logits(1024, 512, dtype, magnitude=2.0, seed=2)
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


def test_distinct_from_default_moe_routing_op():
    """The sibling one flag away: same ids, weights that do not sum to 1."""
    logits = _logits(512, 512, torch.bfloat16, seed=3)
    ids, weights = op(logits, 10, torch.float32)
    other_ids, other_weights = torch.ops.trtllm.default_moe_routing_op(logits, 10, torch.float32)
    assert torch.equal(ids, other_ids)
    row_sums = other_weights.sum(-1)
    assert float(row_sums.max()) < 0.99
    assert (weights - other_weights).abs().max().item() > 1e-2
    torch.testing.assert_close(
        other_weights / row_sums.unsqueeze(-1), weights, rtol=1e-6, atol=1e-7
    )


def test_ties_break_toward_lower_expert_index():
    """Opposite to torch.topk, and common at 512 bf16 experts."""
    logits = torch.zeros(4, 64, dtype=torch.bfloat16, device=DEV)
    logits[0, :] = 1.0
    logits[1, 3] = logits[1, 7] = 5.0
    logits[2, ::2] = 2.0
    logits[3, 0] = logits[3, 63] = 1.0
    ids, _ = _check(logits, 4, torch.float32)
    expected = [[0, 1, 2, 3], [3, 7, 0, 1], [0, 2, 4, 6], [0, 63, 1, 2]]
    assert ids.tolist() == expected

    dense = _logits(4096, 512, torch.bfloat16, seed=4)
    ids, _ = _check(dense, 10, torch.float32)
    torch_topk_ids = torch.topk(dense, 10, dim=-1).indices.to(torch.int32)
    disagreeing = int((ids != torch_topk_ids).any(-1).sum())
    selected = torch.gather(dense, 1, ids.long())
    tied = int((selected[:, :-1] == selected[:, 1:]).any(-1).sum())
    print(f"  {tied} of 4096 rows tie inside the top-10; torch.topk disagrees on {disagreeing}")
    assert disagreeing > 0 and tied > dense.shape[0] // 4
    for tokens, num_experts, topk in ((512, 128, 8), (256, 32, 4)):
        _check(_logits(tokens, num_experts, torch.bfloat16, "coarse", seed=5), topk, torch.float32)


def test_slot_order_and_weight_monotonicity():
    logits = _logits(512, 512, torch.bfloat16, seed=6)
    ids, weights = _check(logits, 10, torch.float32)
    selected = torch.gather(logits.float(), 1, ids.long())
    assert bool((selected[:, :-1] >= selected[:, 1:]).all())
    assert bool((weights[:, :-1] >= weights[:, 1:]).all())


def test_output_dtype_defaults_to_fp32_and_casts_after_the_softmax():
    logits = _logits(1024, 512, torch.float32, magnitude=3.0, seed=7)
    default_ids, default_weights = op(logits, 10)
    assert default_weights.dtype == torch.float32
    explicit_ids, explicit_weights = op(logits, 10, torch.float32)
    assert torch.equal(default_ids, explicit_ids) and torch.equal(default_weights, explicit_weights)
    _, bf16_weights = _check(logits, 10, torch.bfloat16)
    order = torch.sort(logits, dim=-1, descending=True, stable=True).indices[:, :10]
    narrowed = torch.gather(logits, 1, order).to(torch.bfloat16).float()
    e = torch.exp(narrowed - narrowed.max(-1, keepdim=True).values)
    cast_first = (e / e.sum(-1, keepdim=True)).to(torch.bfloat16)
    differing = int((bf16_weights != cast_first).sum())
    print(f"  cast-before-softmax disagrees on {differing} of {bf16_weights.numel()}")
    assert differing > bf16_weights.numel() // 10


# ── edges ─────────────────────────────────────────────────────────────────


def test_degenerate_shapes():
    ids, weights = op(_logits(8, 512, torch.bfloat16), 0, torch.float32)
    assert ids.shape == (8, 0) and weights.shape == (8, 0)
    ids, weights = op(torch.empty(0, 512, dtype=torch.bfloat16, device=DEV), 10, torch.float32)
    assert ids.shape == (0, 10) and weights.shape == (0, 10)
    assert ids.dtype == torch.int32 and weights.dtype == torch.float32


def test_input_not_mutated_and_deterministic():
    logits = _logits(4096, 512, torch.bfloat16, seed=10)
    snapshot = logits.clone()
    ids_a, weights_a = op(logits, 10, torch.bfloat16)
    ids_b, weights_b = op(logits, 10, torch.bfloat16)
    assert torch.equal(logits, snapshot)
    assert torch.equal(ids_a, ids_b) and torch.equal(weights_a, weights_b)
    assert weights_a.data_ptr() not in (weights_b.data_ptr(), logits.data_ptr())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        ids_c, weights_c = op(logits, 10, torch.bfloat16)
    stream.synchronize()
    assert torch.equal(ids_a, ids_c) and torch.equal(weights_a, weights_c)


def test_non_finite_logits_poison_the_whole_row():
    logits = _logits(3, 512, torch.float32)
    logits[0, 5], logits[1, 7] = float("inf"), float("nan")
    logits[2, :] = float("-inf")
    ids, weights = op(logits, 10, torch.float32)
    assert bool(torch.isnan(weights).all())
    assert int(ids[0, 0]) == 5 and int(ids[1, 0]) == 7
    # -inf is a soft mask: excluded while topk fits in the finite experts...
    masked = _logits(4, 512, torch.float32, seed=1)
    masked[:, :8] = float("-inf")
    masked_ids, masked_weights = op(masked, 10, torch.float32)
    assert bool(torch.isfinite(masked_weights).all()) and int(masked_ids.min()) >= 8
    # ...and at weight exactly 0 once it does not
    starved = _logits(4, 512, torch.float32, seed=2)
    starved[:, 7:] = float("-inf")
    starved_ids, starved_weights = op(starved, 10, torch.float32)
    assert bool(torch.isfinite(starved_weights).all())
    assert bool((starved_weights[:, 7:] == 0).all()) and bool((starved_weights[:, :7] > 0).all())
    assert int(starved_ids.max()) < 512


def test_op_rejects_unsupported_domains():
    """Loud on the op's own; none of these needs a guard."""
    logits = _logits(8, 512, torch.bfloat16)
    cases = [
        ("1D router_logits", lambda: op(logits[0], 6)),
        ("3D router_logits", lambda: op(logits.unsqueeze(0), 6)),
        ("fp64 router_logits", lambda: op(logits.double(), 6)),
        ("int32 router_logits", lambda: op(logits.int(), 6)),
        ("fp16 output_dtype", lambda: op(logits, 6, torch.float16)),
        ("fp16 output_dtype, fp16 logits", lambda: op(logits.half(), 6, torch.float16)),
        ("fp64 output_dtype", lambda: op(logits, 6, torch.float64)),
        ("int32 output_dtype", lambda: op(logits, 6, torch.int32)),
        ("topk 17", lambda: op(logits, 17)),
        ("negative topk", lambda: op(logits, -1)),
        ("num_experts 513", lambda: op(_logits(8, 513, torch.bfloat16), 6)),
        ("cpu router_logits", lambda: op(logits.cpu(), 6)),
    ]
    for tag, call in cases:
        with pytest.raises((RuntimeError, ValueError, NotImplementedError)):
            call()


# ── the guards ────────────────────────────────────────────────────────────


def test_unguarded_hazards_are_silent():
    """What `is_valid` exists for: the raw op answers both of these."""
    wide = _logits(8, 1024, torch.bfloat16, seed=12)
    sliced = wide[:, :512]
    dense = wide.reshape(-1)[: 8 * 512].reshape(8, 512).contiguous()
    strided_ids, _ = op(sliced, 10, torch.float32)
    assert torch.equal(strided_ids, op(dense, 10, torch.float32)[0]), (
        "the strided read is no longer a dense reinterpretation"
    )
    assert not torch.equal(strided_ids, op.reference(sliced, 10, torch.float32)[0])
    narrow = _logits(4, 8, torch.bfloat16, seed=13)
    padded_ids, padded_weights = op(narrow, 12, torch.float32)
    assert bool((padded_ids[:, 8:] == 0).all()) and bool((padded_weights[:, 8:] == 0).all())


def test_guards_refuse():
    wide = _logits(8, 1024, torch.bfloat16, seed=12)
    with validating(op):
        with pytest.raises(AssertionError, match="contiguous"):
            op(wide[:, :512], 10)
        with pytest.raises(AssertionError, match="contiguous"):
            op(wide.t(), 6)
        with pytest.raises(AssertionError, match="exceeds num_experts"):
            op(_logits(4, 8, torch.bfloat16), 12)
        _check(wide[:, :512].contiguous(), 10, torch.float32)
