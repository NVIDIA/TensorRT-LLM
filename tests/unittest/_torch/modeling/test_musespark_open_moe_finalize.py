# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-expert post-norm finalize, on shapes small enough to check by hand.

The source normalizes *each selected expert's* output before the top-k
combine. RMSNorm is nonlinear, so normalizing the combined result is a
different function -- which is why the fused MoE kernel runs with
``do_finalize=False`` and this code finalizes instead.

Under expert parallelism each rank owns a slice of the experts, and
``expanded_idx_to_permuted_idx`` holds ``-1`` for every slot whose expert is
not local. The kernel never writes those rows of ``gemm2_output``, so they can
hold anything, including NaN from a previous step's scratch. They have to be
*selected out*, not multiplied by zero: ``0 * NaN`` is NaN, and one such row
poisons the token, then the all-reduce, then every rank.

The all-invalid case is the one the file below is really for. A rank whose
experts none of the batch's tokens picked -- routine at EP4 with a small batch
-- must contribute exact zeros to the all-reduce, not NaN and not
``rsqrt(0 + eps)`` noise.

Complements the finalize tests in ``test_modeling_musespark_open.py``, which pin
the norm-before-combine ordering against a golden transcribed from the source
formula.
"""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open import (
    post_expert_norm_combine,
    post_expert_norm_rows,
)

EPS = 1e-6
HIDDEN = 8
TOP_K = 2


def _rmsnorm_weight_first(row: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The source's form: scale by ``1 + w`` first, then normalize."""
    scaled = row.double() * (weight.double() + 1.0)
    return scaled * torch.rsqrt(scaled.pow(2).mean() + eps)


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(HIDDEN) * 0.1


def _gemm2(num_rows: int, *, fill=None) -> torch.Tensor:
    torch.manual_seed(1)
    out = torch.randn(num_rows, HIDDEN)
    if fill is not None:
        out.fill_(fill)
    return out


def test_no_local_experts_on_this_rank_contributes_exact_zeros(weight):
    """Every slot is -1, and every unwritten row is NaN.

    This is a whole rank's contribution to the all-reduce. It must be exactly
    zero: a NaN here is indistinguishable downstream from a real kernel fault,
    and it arrives on all ranks at once because the all-reduce spreads it.
    """
    num_tokens = 3
    gemm2 = _gemm2(num_tokens * TOP_K, fill=float("nan"))
    index = torch.full((num_tokens * TOP_K,), -1, dtype=torch.int32)
    routing = torch.rand(num_tokens, TOP_K)

    rows, valid = post_expert_norm_rows(
        gemm2, index, weight, EPS, num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN
    )
    combined = post_expert_norm_combine(
        gemm2, index, routing, weight, EPS, num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN
    )

    assert not valid.any()
    assert torch.equal(rows, torch.zeros_like(rows))
    assert torch.equal(combined, torch.zeros_like(combined))
    assert torch.isfinite(combined).all()


def test_a_nan_row_behind_an_invalid_slot_does_not_reach_the_output(weight):
    """The mixed case: this rank owns one of the two experts each token picked."""
    num_tokens = 2
    gemm2 = _gemm2(num_tokens * TOP_K)
    index = torch.tensor([0, -1, 2, -1], dtype=torch.int32)
    gemm2[1].fill_(float("nan"))
    gemm2[3].fill_(float("inf"))
    routing = torch.tensor([[0.7, 0.3], [0.4, 0.6]])

    combined = post_expert_norm_combine(
        gemm2, index, routing, weight, EPS, num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN
    )

    assert torch.isfinite(combined).all()
    for token, (row, w) in enumerate(zip((0, 2), (0.7, 0.4))):
        expected = _rmsnorm_weight_first(gemm2[row], weight, EPS) * w
        torch.testing.assert_close(combined[token].double(), expected, rtol=1e-6, atol=1e-6)


def test_the_combine_is_the_weighted_sum_of_the_rows(weight):
    """``post_expert_norm_rows`` is exactly what the combine sums.

    They are separate functions so the bring-up can compare the individual
    post-norm rows rather than only their sum -- against a sum, a
    norm-after-combine bug and a norm-before-combine implementation look the
    same.
    """
    num_tokens = 4
    gemm2 = _gemm2(num_tokens * TOP_K)
    index = torch.arange(num_tokens * TOP_K, dtype=torch.int32)
    index[3] = -1
    routing = torch.rand(num_tokens, TOP_K)

    rows, _ = post_expert_norm_rows(
        gemm2, index, weight, EPS, num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN
    )
    combined = post_expert_norm_combine(
        gemm2, index, routing, weight, EPS, num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN
    )

    expected = (rows * routing.reshape(num_tokens, TOP_K, 1)).sum(dim=1)
    torch.testing.assert_close(combined, expected)


def test_normalizing_after_the_combine_is_a_different_function(weight):
    """Guards the reason ``do_finalize=False`` is required of the backend."""
    num_tokens = 2
    gemm2 = _gemm2(num_tokens * TOP_K)
    index = torch.arange(num_tokens * TOP_K, dtype=torch.int32)
    routing = torch.tensor([[0.6, 0.4], [0.5, 0.5]])

    before = post_expert_norm_combine(
        gemm2, index, routing, weight, EPS, num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN
    )

    raw = gemm2.reshape(num_tokens, TOP_K, HIDDEN).float()
    summed = (raw * routing.reshape(num_tokens, TOP_K, 1)).sum(dim=1)
    scaled = summed * (weight.float() + 1.0)
    after = scaled * torch.rsqrt(scaled.pow(2).mean(-1, keepdim=True) + EPS)

    assert not torch.allclose(before, after, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_the_finalize_accumulates_in_fp32(weight, dtype):
    """Whatever the kernel's output dtype, the norm and the sum run in FP32.

    The 62-layer residual stream is where a bf16 combine would show up, and it
    is the cheapest place to lose accuracy for nothing.
    """
    num_tokens = 2
    gemm2 = _gemm2(num_tokens * TOP_K).to(dtype)
    index = torch.arange(num_tokens * TOP_K, dtype=torch.int32)
    routing = torch.rand(num_tokens, TOP_K).to(dtype)

    combined = post_expert_norm_combine(
        gemm2,
        index,
        routing,
        weight.to(dtype),
        EPS,
        num_tokens=num_tokens,
        top_k=TOP_K,
        hidden_size=HIDDEN,
    )

    assert combined.dtype is torch.float32
    assert combined.shape == (num_tokens, HIDDEN)


def test_a_wider_gemm2_row_is_truncated_to_the_expert_hidden_size(weight):
    """The kernel's rows are padded; only the first ``hidden_size`` count."""
    num_tokens = 2
    padded = torch.cat(
        [_gemm2(num_tokens * TOP_K), torch.full((num_tokens * TOP_K, 5), float("nan"))], dim=1
    )
    index = torch.arange(num_tokens * TOP_K, dtype=torch.int32)
    routing = torch.rand(num_tokens, TOP_K)

    combined = post_expert_norm_combine(
        padded, index, routing, weight, EPS, num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN
    )

    assert torch.isfinite(combined).all()


def test_the_norm_is_applied_on_every_layer_including_bf16_ones(weight):
    """No dtype-conditional shortcut in the finalize.

    One reference runtime skips this per-expert post-norm on the two BF16
    layers (0 and 61) unless a specific backend is selected. That is a bug in
    the reference, not a property of the model: the finalize here is
    dtype-blind, and a BF16 expert output is normalized exactly like an NVFP4
    one. Checked by feeding the same values in both dtypes and requiring the
    same answer to BF16 tolerance.
    """
    num_tokens = 2
    fp32 = _gemm2(num_tokens * TOP_K)
    bf16 = fp32.to(torch.bfloat16)
    index = torch.arange(num_tokens * TOP_K, dtype=torch.int32)
    routing = torch.tensor([[0.6, 0.4], [0.5, 0.5]])

    kwargs = dict(num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN)
    from_fp32 = post_expert_norm_combine(fp32, index, routing, weight, EPS, **kwargs)
    from_bf16 = post_expert_norm_combine(bf16, index, routing, weight, EPS, **kwargs)

    # Non-zero, so "both skipped the norm" cannot pass this.
    assert from_fp32.abs().max() > 0.1
    torch.testing.assert_close(from_bf16, from_fp32, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("ep_size", [2, 4])
def test_finalizing_per_rank_then_summing_equals_finalizing_the_whole_set(weight, ep_size):
    """Why the finalize is allowed to run *before* the cross-rank combine.

    Under attention DP the MoE layer acquires a ``Communication.combine``
    between the expert GEMM and this module, and no combine can carry the
    unfinalized triple: they are all typed for a dense per-token tensor. The
    way out is to finalize the dispatched rows first and let the combine
    reduce already-finalized partials -- which is only valid if that reordering
    is the same function.

    It is, because ``w * RMSNorm(gain * y)`` is computed from one
    ``(token, expert)`` row alone and a non-local slot contributes an exact
    zero. So a sum of per-rank partial sums is the whole sum, up to summation
    order. Checked here by splitting the experts across ranks and filling every
    row a rank does not own with NaN, which is what the kernel actually leaves
    behind: it is the reordering AND the NaN-selection that have to hold
    together, since a rank that multiplied a non-local row by zero would poison
    the reduction instead of the slot.
    """
    num_tokens, num_experts = 6, 8
    torch.manual_seed(3)
    routing = torch.rand(num_tokens, TOP_K)
    chosen = torch.stack([torch.randperm(num_experts)[:TOP_K] for _ in range(num_tokens)])
    kwargs = dict(num_tokens=num_tokens, top_k=TOP_K, hidden_size=HIDDEN)

    all_rows = _gemm2(num_tokens * TOP_K)
    all_index = torch.arange(num_tokens * TOP_K, dtype=torch.int32).reshape(num_tokens, TOP_K)
    whole = post_expert_norm_combine(all_rows, all_index, routing, weight, EPS, **kwargs)

    partials = []
    for ep_rank in range(ep_size):
        per_rank = num_experts // ep_size
        local = (chosen >= ep_rank * per_rank) & (chosen < (ep_rank + 1) * per_rank)
        rank_rows = torch.full_like(all_rows, float("nan"))
        rank_rows[local.reshape(-1)] = all_rows[local.reshape(-1)]
        rank_index = torch.where(local, all_index, torch.full_like(all_index, -1))
        partials.append(
            post_expert_norm_combine(rank_rows, rank_index, routing, weight, EPS, **kwargs)
        )

    summed = torch.stack(partials).sum(dim=0)
    assert torch.isfinite(summed).all()
    assert whole.abs().max() > 0.1
    torch.testing.assert_close(summed, whole, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# The fused CUDA path (lever 1c).
#
# Everything above runs the reference composition, because ``post_expert_norm_
# combine`` only dispatches to the Triton kernel on CUDA. These re-check the
# same contracts against the fused kernel, at the checkpoint's real width as
# well as the by-hand one, and pin the two facts that only the fused path can
# get wrong: that a non-local slot's row is never *read* (so a NaN behind it
# cannot reach the accumulator), and that the kernel is CUDA-graph safe.
# --------------------------------------------------------------------------

CKPT_HIDDEN = 4096  # expert_hidden_size, i.e. override_routed_hidden_dim
CKPT_TOP_K = 8

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused finalize is CUDA-only")


def _reference_combine(gemm2, index, routing, weight, eps, *, num_tokens, top_k, hidden_size):
    """The op-by-op composition, i.e. what the fused kernel has to reproduce."""
    rows, _ = post_expert_norm_rows(
        gemm2, index, weight, eps, num_tokens=num_tokens, top_k=top_k, hidden_size=hidden_size
    )
    return (rows * routing.reshape(num_tokens, top_k, 1).to(torch.float32)).sum(dim=1)


def _decode_case(
    num_tokens,
    *,
    hidden=CKPT_HIDDEN,
    top_k=CKPT_TOP_K,
    dtype=torch.bfloat16,
    invalid=(),
    pad=0,
    seed=0,
):
    """A decode-shaped finalize input, with ``invalid`` flat slots set to -1.

    Every row behind an invalid slot is filled with NaN, which is what the MoE
    kernel actually leaves there: it never writes the rows of experts that are
    not local to this rank.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    slots = num_tokens * top_k
    gemm2 = torch.randn(slots, hidden + pad, generator=gen, device="cuda", dtype=torch.float32)
    index = torch.arange(slots, dtype=torch.int32, device="cuda")
    for slot in invalid:
        index[slot] = -1
        gemm2[slot].fill_(float("nan"))
    if pad:
        gemm2[:, hidden:].fill_(float("nan"))
    routing = torch.rand(num_tokens, top_k, generator=gen, device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden, generator=gen, device="cuda", dtype=torch.float32) * 0.1
    return gemm2.to(dtype), index, routing, weight.to(dtype)


@cuda_only
@pytest.mark.parametrize("num_tokens", [1, 7, 256])
def test_fused_matches_the_reference_at_the_checkpoint_width(num_tokens):
    """Rounding-different, declared in advance -- but far below the BF16 grain.

    The fused reduction sums the squares and the top-k in a different order, so
    this is not bit-exact and was never expected to be. What it must be is
    negligible against the dtype the result lands in: ``combined`` is cast to
    BF16 two statements after the all-reduce, and BF16's relative grain is
    ~3.9e-3.
    """
    gemm2, index, routing, weight = _decode_case(num_tokens, invalid=(0, 3))
    kwargs = dict(num_tokens=num_tokens, top_k=CKPT_TOP_K, hidden_size=CKPT_HIDDEN)

    fused = post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)
    reference = _reference_combine(gemm2, index, routing, weight, EPS, **kwargs)

    assert torch.isfinite(fused).all()
    max_abs = (fused - reference).abs().max().item()
    scale = reference.abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        fused.reshape(1, -1).double(), reference.reshape(1, -1).double()
    ).item()
    print(f"T={num_tokens} max_abs={max_abs:.3e} rel={max_abs / scale:.3e} cosine={cosine:.12f}")
    assert max_abs < 1e-4 * scale
    assert cosine > 1 - 1e-9
    torch.testing.assert_close(fused, reference, rtol=2e-5, atol=2e-5)


@cuda_only
def test_fused_never_reads_a_row_behind_an_invalid_slot():
    """The whole-rank case, at the real width: exact zeros, not NaN, not noise.

    Every slot is -1 and every row is NaN. This rank's contribution to the
    all-reduce has to be exactly zero -- a NaN here is spread to every rank by
    the reduce and is indistinguishable downstream from a real kernel fault.
    """
    num_tokens = 16
    gemm2, index, routing, weight = _decode_case(
        num_tokens, invalid=tuple(range(num_tokens * CKPT_TOP_K))
    )

    combined = post_expert_norm_combine(
        gemm2,
        index,
        routing,
        weight,
        EPS,
        num_tokens=num_tokens,
        top_k=CKPT_TOP_K,
        hidden_size=CKPT_HIDDEN,
    )

    assert torch.equal(combined, torch.zeros_like(combined))


@cuda_only
def test_fused_selects_out_a_permuted_index_past_the_end_of_gemm2():
    """A slot whose permuted index addresses past the buffer contributes zero.

    The -1 guard covers the low end -- the backend's marker for a non-local
    slot. This is the high end, and it is not symmetry for its own sake:
    ``expanded_idx_to_permuted_idx`` is a function of how the WHOLE batch
    routed, so an index past the end would be a batch-composition-dependent
    read of memory this kernel does not own. It would not fault; it would
    return whatever the allocator last left there and surface as a numerics
    divergence that appears only at some batch compositions -- which is exactly
    the signature that is hard to tell from a real accuracy regression.

    Triton masks only what it is told to mask, so the bound has to be explicit.
    The check is that an out-of-range slot behaves identically to a -1 slot: it
    is selected out, and the token is the weighted sum of its remaining slots.
    """
    num_tokens, slots = 4, 4 * CKPT_TOP_K
    gemm2, index, routing, weight = _decode_case(num_tokens)
    kwargs = dict(num_tokens=num_tokens, top_k=CKPT_TOP_K, hidden_size=CKPT_HIDDEN)

    # Two out-of-range values that a corrupted or mis-sized index could take:
    # exactly one past the end, and far past it.
    out_of_range = index.clone()
    out_of_range[1] = slots
    out_of_range[5] = slots + 1024

    # The reference: the same two slots marked -1, which the kernel already
    # selects out. If the upper bound works, the two agree exactly.
    as_invalid = index.clone()
    as_invalid[1] = -1
    as_invalid[5] = -1

    got = post_expert_norm_combine(gemm2, out_of_range, routing, weight, EPS, **kwargs)
    want = post_expert_norm_combine(gemm2, as_invalid, routing, weight, EPS, **kwargs)

    assert torch.isfinite(got).all(), "an out-of-range slot leaked a non-finite value"
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@cuda_only
def test_fused_ignores_the_padding_columns_of_a_wider_gemm2_row():
    """The kernel's rows are padded and the padding can hold anything."""
    num_tokens = 4
    gemm2, index, routing, weight = _decode_case(num_tokens, pad=64)
    kwargs = dict(num_tokens=num_tokens, top_k=CKPT_TOP_K, hidden_size=CKPT_HIDDEN)

    fused = post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)

    assert torch.isfinite(fused).all()
    torch.testing.assert_close(
        fused,
        _reference_combine(gemm2, index, routing, weight, EPS, **kwargs),
        rtol=2e-5,
        atol=2e-5,
    )


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_accumulates_in_fp32_whatever_the_expert_output_dtype(dtype):
    num_tokens = 8
    gemm2, index, routing, weight = _decode_case(num_tokens, hidden=256, top_k=4, dtype=dtype)
    kwargs = dict(num_tokens=num_tokens, top_k=4, hidden_size=256)

    fused = post_expert_norm_combine(gemm2, index, routing.to(dtype), weight, EPS, **kwargs)

    assert fused.dtype is torch.float32
    torch.testing.assert_close(
        fused,
        _reference_combine(gemm2, index, routing.to(dtype), weight, EPS, **kwargs),
        rtol=2e-5,
        atol=2e-5,
    )


@cuda_only
def test_fused_matches_the_by_hand_shapes_the_cpu_tests_use():
    """Same shapes as the reference tests above, on the fused path.

    HIDDEN=8 exercises the masked tail of ``BLOCK_H``: eight live lanes in a
    block sized to the next power of two, with the rest masked out of both the
    load and the variance.
    """
    torch.manual_seed(0)
    weight = (torch.randn(HIDDEN) * 0.1).cuda()
    gemm2 = torch.randn(4 * TOP_K, HIDDEN).cuda()
    index = torch.arange(4 * TOP_K, dtype=torch.int32).cuda()
    index[3] = -1
    gemm2[3].fill_(float("nan"))
    routing = torch.rand(4, TOP_K).cuda()
    kwargs = dict(num_tokens=4, top_k=TOP_K, hidden_size=HIDDEN)

    fused = post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)

    assert torch.isfinite(fused).all()
    torch.testing.assert_close(
        fused,
        _reference_combine(gemm2, index, routing, weight, EPS, **kwargs),
        rtol=2e-5,
        atol=2e-5,
    )


@cuda_only
def test_fused_finalize_replays_from_a_cuda_graph():
    """The serving path captures the decode step; a JIT compile inside a capture
    is illegal, and a kernel that silently fell back would void the hard-path
    evidence. Warm up once (which is what the runtime's warmup steps do), then
    capture, then replay against fresh inputs.
    """
    num_tokens = 32
    gemm2, index, routing, weight = _decode_case(num_tokens, invalid=(1, 2))
    kwargs = dict(num_tokens=num_tokens, top_k=CKPT_TOP_K, hidden_size=CKPT_HIDDEN)

    post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    static_out = None
    with torch.cuda.graph(graph):
        static_out = post_expert_norm_combine(gemm2, index, routing, weight, EPS, **kwargs)

    fresh_gemm2, _, fresh_routing, _ = _decode_case(num_tokens, invalid=(1, 2), seed=5)
    gemm2.copy_(fresh_gemm2)
    routing.copy_(fresh_routing)
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_combine(gemm2, index, routing, weight, EPS, **kwargs)
    assert torch.isfinite(static_out).all()
    torch.testing.assert_close(static_out, expected, rtol=2e-5, atol=2e-5)
