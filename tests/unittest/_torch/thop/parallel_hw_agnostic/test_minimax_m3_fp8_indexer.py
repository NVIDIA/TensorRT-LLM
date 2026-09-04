# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch


def _reference(qk, num_heads_q, q_weight, k_weight, position_ids):
    reference = qk.clone()
    torch.ops.trtllm.fused_qk_norm_rope(
        reference,
        num_heads_q,
        1,
        0,
        128,
        64,
        1e-5,
        q_weight,
        k_weight,
        10000.0,
        True,  # is_neox
        position_ids,
        1.0,
        0.0,
        0.0,
        1.0,
        True,  # is_qk_norm
        True,  # use_gemma
        False,  # use_mrope
        0,
        0,
    )
    q, k = reference.split([num_heads_q * 128, 128], dim=-1)
    return q.view(q.shape[0], num_heads_q, 128).to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)


def _strided_cache(num_pages, page_size=128, stride_scale=7):
    backing = torch.zeros(
        num_pages * stride_scale,
        1,
        page_size,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    return backing[::stride_scale]


def _guarded_cache(num_pages, page_size=128, stride_scale=7, guard_pages=2):
    """A strided cache view with zeroed guard pages on both sides of it.

    Returns (view, below, above). A store from a negative slot lands in `below`,
    one from a slot past the last page lands in `above`, so tests can assert on
    inspectable memory rather than wait for a fault that may never come. Page
    stride matches _strided_cache, keeping the non-contiguous page axis the op
    has to support.
    """
    total_pages = num_pages + 2 * guard_pages
    backing = torch.zeros(
        total_pages * stride_scale,
        1,
        page_size,
        128,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    first = guard_pages * stride_scale
    last = (guard_pages + num_pages) * stride_scale
    view = backing[first:last:stride_scale]
    assert view.shape[0] == num_pages
    return view, backing[:first], backing[last:]


def _assert_untouched(*regions):
    for region in regions:
        assert torch.count_nonzero(region.view(torch.uint8)).item() == 0


def _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q=4):
    return torch.ops.trtllm.minimax_m3_fp8_indexer_qk_norm_rope(
        qk,
        cache,
        slots,
        num_heads_q,
        128,
        64,
        1e-5,
        q_weight,
        k_weight,
        10000.0,
        position_ids,
    )


@pytest.mark.parametrize("num_tokens", [1, 16, 129])
def test_minimax_m3_fp8_indexer_matches_bf16_then_cast(num_tokens):
    torch.manual_seed(1234)
    num_heads_q = 4
    page_size = 128
    qk = torch.randn(
        num_tokens,
        (num_heads_q + 1) * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda") + 8192
    within = torch.arange(num_tokens, dtype=torch.int32, device="cuda") % page_size
    pages = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    slots = pages * page_size + within
    cache = _strided_cache(num_tokens)

    q_out = _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)
    q_ref, k_ref = _reference(qk, num_heads_q, q_weight, k_weight, position_ids)
    k_out = cache[pages.long(), 0, within.long()]

    # The specialized kernel uses powf while fused_qk_norm_rope uses the
    # exp2f/log2f equivalent, so values at an FP8 boundary can round to
    # adjacent E4M3 values.
    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=0.13, atol=0.05)
    torch.testing.assert_close(k_out.float(), k_ref.float(), rtol=0.13, atol=0.05)


def test_minimax_m3_fp8_indexer_cuda_graph_replay_updates_outputs():
    torch.manual_seed(5678)
    num_tokens = 16
    num_heads_q = 4
    qk = torch.randn(
        num_tokens,
        (num_heads_q + 1) * 128,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda") + 4096
    pages = torch.arange(num_tokens, dtype=torch.int32, device="cuda")
    within = (pages * 11) % 128
    slots = pages * 128 + within
    cache = _strided_cache(num_tokens)

    for _ in range(3):
        _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q_out = _run(qk, cache, slots, q_weight, k_weight, position_ids, num_heads_q)

    first_q = q_out.clone()
    qk.copy_(torch.randn_like(qk))
    graph.replay()
    torch.cuda.synchronize()
    q_ref, k_ref = _reference(qk, num_heads_q, q_weight, k_weight, position_ids)
    k_out = cache[pages.long(), 0, within.long()]

    assert not torch.equal(q_out.view(torch.uint8), first_q.view(torch.uint8))
    torch.testing.assert_close(q_out.float(), q_ref.float(), rtol=0.13, atol=0.05)
    torch.testing.assert_close(k_out.float(), k_ref.float(), rtol=0.13, atol=0.05)


def _run_with_slot_tail(tail_slot, num_live_tokens, seed):
    """Run the kernel over a padded token extent whose tail carries tail_slot.

    Mirrors what the model hands over: index-Q is produced for every row, while
    only the live prefix owns a cache slot. Asserts that the tail wrote nothing
    anywhere and that the live rows still match the reference. Each live row
    takes a page of its own, so num_live_tokens must not exceed num_pages.
    """
    torch.manual_seed(seed)
    num_heads_q, page_size, num_pages, padded_tokens = 4, 128, 4, 17

    qk = torch.randn(padded_tokens, (num_heads_q + 1) * 128, dtype=torch.bfloat16, device="cuda")
    q_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    k_weight = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    position_ids = torch.arange(padded_tokens, dtype=torch.int32, device="cuda") + 1024

    view, below, above = _guarded_cache(num_pages, page_size)
    slots = torch.full((padded_tokens,), tail_slot, dtype=torch.int32, device="cuda")
    pages = torch.arange(num_live_tokens, dtype=torch.int32, device="cuda")
    within = (pages * 37) % page_size
    slots[:num_live_tokens] = pages * page_size + within

    q_out = _run(qk, view, slots, q_weight, k_weight, position_ids, num_heads_q)
    torch.cuda.synchronize()

    _assert_untouched(below, above)

    # The live rows still land where they should, with the right values.
    q_ref, k_ref = _reference(qk, num_heads_q, q_weight, k_weight, position_ids)
    torch.testing.assert_close(
        q_out[:num_live_tokens].float(),
        q_ref[:num_live_tokens].float(),
        rtol=0.13,
        atol=0.05,
    )
    if num_live_tokens:
        torch.testing.assert_close(
            view[pages.long(), 0, within.long()].float(),
            k_ref[:num_live_tokens].float(),
            rtol=0.13,
            atol=0.05,
        )

    # Catches a tail store that happens to land on a valid page.
    live = torch.zeros(num_pages, page_size, dtype=torch.bool, device="cuda")
    live[pages.long(), within.long()] = True
    _assert_untouched(view[:, 0][~live])


@pytest.mark.parametrize("num_live_tokens", [0, 1, 4])
def test_minimax_m3_fp8_indexer_skips_negative_slots(num_live_tokens):
    """A negative slot marks a padded row and must be a cache-write no-op.

    Piecewise CUDA graphs pad token-shaped inputs without adding requests, so
    the producer takes its token count from the padded height and the sentinel
    tail of msa_out_cache_loc reaches the kernel. Unguarded, every padded row in
    every sparse layer stores to the same bytes below the cache base.
    """
    _run_with_slot_tail(-1, num_live_tokens, seed=24)


@pytest.mark.parametrize("tail_slot", [4 * 128, 4 * 128 + 61, 5 * 128 + 127])
def test_minimax_m3_fp8_indexer_skips_slots_past_the_cache(tail_slot):
    """A slot past the last page must not scatter outside the pool.

    out_cache_loc is staged in uninitialized buffers and gathered through an
    index clamped to the staging width rather than a request's own length, so a
    stale or garbage slot is reachable. The parameters start at the first
    out-of-range slot and stay inside the guard pages, so a regression fails an
    assert instead of killing the process.
    """
    _run_with_slot_tail(tail_slot, num_live_tokens=4, seed=25)
