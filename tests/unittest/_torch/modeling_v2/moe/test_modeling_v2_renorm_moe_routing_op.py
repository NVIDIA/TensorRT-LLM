# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the renorm_moe_routing_op catalog entry.

The entry owns the contract. This file turns a cell into logits and holds the
claims no cell carries: the return order, the tie-break against `torch.topk`,
the sibling op one flag away, the sweep of the old certified surface, the edge
behaviour, and the guards. The Mistral Large 4 cells add three spec keys:
`graph_tokens` (each batch captured in a CUDA graph and replayed),
`tokens_range` (every row count of the range, each its own call) and
`rank_tokens` (the dep4 gathered layout: each rank's live rows zero-padded to
`tokens / len(rank_tokens)`). The dep8 cell drives `graph_tokens` at the
decode graphs' 8-rank gathered batches.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op as op,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "renorm_moe_routing_op requires a CUDA device"


def _compare(
    out: tuple[torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Ids equal and weights bit-exact.

    The reference is not an approximation of the kernel: over 2040
    configurations, not one id and not one weight bit differed; at 128 / top-4
    on fp32 logits, none differed at any T in 1..131072 nor at 9216, 32768 or
    131072 tokens for logits of sigma 1e-3 to 100, in either output dtype, nor
    on the looping path past the 8192-block cap (T 32769..131073). A one-ulp
    `assert_close` runs first only so a drift reports a readable diff.
    The nearest wrong answers sit 0.4476 away (un-renormalized softmax,
    `1/topk`), and a `torch.topk` reference differs on a third of the rows at
    512 bf16 experts. At 128 / top-4, fp32 logits ranked after narrowing to
    bf16 move the ids of 566 of 9216 rows and 28394 of 36864 bf16 weights.
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


def _bits(weights: torch.Tensor) -> torch.Tensor:
    """The weights' bit patterns, so `!=` is bitwise (NaN and -0.0 included)."""
    return weights.view(torch.int16 if weights.element_size() == 2 else torch.int32)


def _cell_logits(s, seed, tokens=None):
    return _logits(
        s["tokens"] if tokens is None else tokens,
        s["num_experts"],
        s["logits_dtype"],
        s["dist"],
        s.get("magnitude", 1.0),
        seed=seed,
    )


def _words(x: torch.Tensor) -> torch.Tensor:
    """`x`'s bits as int64 words where a row is whole words (at top-4: the ids
    two words a row, the bf16 weights one), else as `_bits`."""
    return x.view(torch.int64) if x.shape[-1] * x.element_size() % 8 == 0 else _bits(x)


_SWEEP_GROUP = 16


def _sweep(s, seed) -> None:
    """Every T of `tokens_range`, each its own call, on the leading T rows of one set.

    The reference is taken once, at the top of the range, and sliced: a
    token's routing reads only its own row, so the leading T rows of the
    reference are the reference for a T-row call. The gate is `_compare`'s,
    ids equal and weights bitwise, read as int64 words: the outputs of
    `_SWEEP_GROUP` consecutive T are concatenated and compared with the
    matching reference rows at once, the verdict flagged on the device, so
    the sweep pays one synchronization and few compare launches. A flag per T
    cost 1.87 s over 1..32768 and 7.03 s over 32769..131072; a flag per group
    of 16, 0.88 s and 4.55 s, with the op alone 4.18 s over 32769..131072.
    One flipped weight bit in one T of a group is seen. Each T of the first
    failing group is re-driven through `_compare` for its diff.
    """
    lo, hi = s["tokens_range"]
    logits = _cell_logits(s, seed, tokens=hi)
    ref_ids, ref_weights = op.reference(logits, s["topk"], s["output_dtype"])
    ref_id_words, ref_weight_words = _words(ref_ids), _words(ref_weights)
    row_counts = range(lo, hi + 1)
    groups = [row_counts[i : i + _SWEEP_GROUP] for i in range(0, len(row_counts), _SWEEP_GROUP)]
    bad = torch.zeros(len(groups), dtype=torch.bool, device=DEV)
    for g, group in enumerate(groups):
        outs = [op(logits[:t], s["topk"], s["output_dtype"]) for t in group]
        for t, (ids, weights) in zip(group, outs):
            assert ids.shape == weights.shape == (t, s["topk"])
            assert weights.dtype is s["output_dtype"]
        torch.logical_or(
            torch.ne(
                torch.cat([_words(ids) for ids, _ in outs]),
                torch.cat([ref_id_words[:t] for t in group]),
            ).any(),
            torch.ne(
                torch.cat([_words(weights) for _, weights in outs]),
                torch.cat([ref_weight_words[:t] for t in group]),
            ).any(),
            out=bad[g],
        )
    failing = torch.nonzero(bad).flatten()
    if failing.numel():
        for t in groups[int(failing[0])]:
            out = op(logits[:t], s["topk"], s["output_dtype"])
            _compare(out, (ref_ids[:t], ref_weights[:t]))
    assert failing.numel() == 0, (
        f"{failing.numel()} groups of {_SWEEP_GROUP} row counts differ from the reference; "
        f"first T={groups[int(failing[0])][0]}..{groups[int(failing[0])][-1]}"
    )


def _replays(s, seed) -> None:
    """Each captured batch in its own CUDA graph, replayed on new logits.

    The replay has to equal an eager call on the same logits bit for bit -- a
    graph that froze something of the capture would differ -- and is gated
    against the reference like any other output. Each capture is preceded by
    one call on the capture stream, as an engine's warm-up is.
    """
    stream = torch.cuda.Stream()
    for t in s["graph_tokens"]:
        static = _cell_logits(s, seed + t, tokens=t)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op(static, s["topk"], s["output_dtype"])
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ids, weights = op(static, s["topk"], s["output_dtype"])
        for replay in range(2):
            fresh = _cell_logits(s, seed + 1000 * (replay + 1) + t, tokens=t)
            static.copy_(fresh)
            graph.replay()
            eager_ids, eager_weights = op(fresh, s["topk"], s["output_dtype"])
            assert torch.equal(ids, eager_ids) and torch.equal(
                _bits(weights), _bits(eager_weights)
            ), f"T={t}: graph replay {replay} differs from eager"
            _compare((ids, weights), op.reference(fresh, s["topk"], s["output_dtype"]))


def _padded(s, seed) -> None:
    """The dep4 gathered layout: rank r's `rank_tokens[r]` live rows, then
    all-+0.0 rows up to T_max = tokens / ranks, the ranks' blocks in order.

    +0.0 in every column is what a zero hidden row gives through the router
    GEMM. The gate is `_compare` against the reference, and every padding row
    must be the exact 128-way tie's answer: ids 0..topk-1 at weight 1/topk.
    """
    t_max = s["tokens"] // len(s["rank_tokens"])
    logits = torch.zeros(s["tokens"], s["num_experts"], dtype=s["logits_dtype"], device=DEV)
    padding = torch.ones(s["tokens"], dtype=torch.bool, device=DEV)
    for rank, count in enumerate(s["rank_tokens"]):
        logits[rank * t_max : rank * t_max + count] = _cell_logits(s, seed + rank, tokens=count)
        padding[rank * t_max : rank * t_max + count] = False
    ids, weights = _check(logits, s["topk"], s["output_dtype"])
    assert int(padding.sum()) == s["tokens"] - sum(s["rank_tokens"])
    tie_ids = torch.arange(s["topk"], dtype=torch.int32, device=DEV)
    assert bool((ids[padding] == tie_ids).all()), "a padding row is not routed to experts 0..topk-1"
    assert bool((weights[padding] == 1.0 / s["topk"]).all()), "a padding weight is not 1/topk"


# ── the certified cells ───────────────────────────────────────────────────


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    s = cell.spec
    seed = zlib.crc32(cell.why.encode())
    if "tokens_range" in s:
        _sweep(s, seed)
        return
    if "graph_tokens" in s:
        _replays(s, seed)
        return
    if "rank_tokens" in s:
        _padded(s, seed)
        return
    logits = _cell_logits(s, seed)
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


def test_top4_of_128_matches_the_softmax_over_all_experts():
    """Mistral Large 4's routing as its reference model states it -- an fp32
    softmax over all 128 experts, the top 4 probabilities, divided by their
    sum -- against the op's top-4-then-softmax, at the target's values.

    Both exponentiate the same differences from the row's largest logit, so
    the two differ only in how the division is split: measured at most
    2.4e-07 relative in fp32 (about 2 ulp; the gate allows 8), and the bf16
    weights bitwise equal on these 9216 rows (the gate allows one bf16
    rounding flip). The same probabilities left un-renormalized sit more than
    1e-2 away.
    """
    logits = _logits(9216, 128, torch.float32, magnitude=3.0, seed=14)
    probs, prob_ids = torch.topk(torch.softmax(logits, dim=-1), 4, dim=-1)
    renormalized = probs / probs.sum(-1, keepdim=True)
    ids, weights = op(logits, 4, torch.float32)
    assert torch.equal(ids, prob_ids.to(torch.int32))
    torch.testing.assert_close(weights, renormalized, rtol=8 * 2.0**-23, atol=0.0)
    assert (weights - probs).abs().max().item() > 1e-2
    bf16_ids, bf16_weights = op(logits, 4, torch.bfloat16)
    assert torch.equal(bf16_ids, ids)
    torch.testing.assert_close(
        bf16_weights.float(), renormalized.to(torch.bfloat16).float(), rtol=2.0**-8, atol=0.0
    )


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


def test_sign_bits_rank_by_their_bits():
    """Where the op and `reference` part: the kernel orders logits by their
    bit pattern, torch's sort by their value.

    +0.0 ranks above -0.0, where torch holds the two zeros equal and keeps the
    lower index first; the weights agree, since both zeros exponentiate to 1.
    A NaN with its sign bit set ranks below -inf, so the op skips it and
    answers the rest of the row as if it held -1e30, where torch ranks it
    first and answers NaN.
    """
    logits = torch.full((2, 128), -0.0, dtype=torch.float32, device=DEV)
    logits[0, 9] = 0.0
    logits[1] = 0.0
    ids, weights = op(logits, 4, torch.bfloat16)
    ref_ids, ref_weights = op.reference(logits, 4, torch.bfloat16)
    assert ids.tolist() == [[9, 0, 1, 2], [0, 1, 2, 3]]
    assert ref_ids.tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert torch.equal(weights, ref_weights) and bool((weights == 0.25).all())

    row = _logits(1, 128, torch.float32, magnitude=3.0, seed=18)
    masked = row.clone()
    masked[0, 5] = -1e30
    row.view(torch.int32)[0, 5] = -1  # 0xffffffff: a NaN with its sign bit set
    assert bool(torch.isnan(row[0, 5]))
    ids, weights = op(row, 4, torch.bfloat16)
    masked_ids, masked_weights = op(masked, 4, torch.bfloat16)
    assert torch.equal(ids, masked_ids) and torch.equal(weights, masked_weights)
    assert 5 not in ids.tolist()[0] and bool(torch.isfinite(weights).all())
    ref_ids, ref_weights = op.reference(row, 4, torch.bfloat16)
    assert int(ref_ids[0, 0]) == 5 and bool(torch.isnan(ref_weights).all())


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


def test_garbage_rows_keep_their_ids_in_range_and_to_themselves():
    """A padded 16-row batch whose last rows hold anything at all.

    Whatever a padding row holds -- random bit patterns, NaN, -inf -- its ids
    are distinct experts in [0, 128), so an expert runner downstream never
    indexes outside its table, and the live rows are bitwise what they are
    without the padding.
    """
    live = _logits(13, 128, torch.float32, magnitude=3.0, seed=15)
    g = torch.Generator(device=DEV).manual_seed(16)
    raw = torch.randint(-(2**31), 2**31, (128,), generator=g, device=DEV, dtype=torch.int64)
    batch = torch.cat(
        [
            live,
            raw.to(torch.int32).view(torch.float32).unsqueeze(0),
            torch.full((1, 128), float("nan"), device=DEV),
            torch.full((1, 128), float("-inf"), device=DEV),
        ]
    )
    ids, weights = op(batch, 4, torch.bfloat16)
    assert int(ids.min()) >= 0 and int(ids.max()) < 128
    assert bool((ids.sort(-1).values.diff(dim=-1) > 0).all()), "a row repeats an expert"
    live_ids, live_weights = _check(live, 4, torch.bfloat16)
    assert torch.equal(ids[:13], live_ids) and torch.equal(_bits(weights[:13]), _bits(live_weights))


def test_a_row_window_is_read_at_its_offset():
    """A contiguous run of rows inside a larger buffer: a nonzero storage
    offset, which `data_ptr()` carries, and the guard admits."""
    logits = _logits(64, 128, torch.float32, magnitude=3.0, seed=17)
    window = logits[17:40]
    assert window.is_contiguous() and window.storage_offset() == 17 * 128
    with validating(op):
        _check(window, 4, torch.bfloat16)


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
