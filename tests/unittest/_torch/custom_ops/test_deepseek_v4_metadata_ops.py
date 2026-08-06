# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the DeepSeek-V4 metadata-preparation CUDA ops.

Each op replaces a python/ATen reference that used to run in
``attn_metadata.prepare``. The references are reproduced verbatim here so the
tests pin the exact semantics rather than the current implementation.
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

RATIO_SETS = [[1, 4, 128], [1, 128], [128], [1, 4], [1]]


# ---------------------------------------------------------------- references


def _ref_indices(token_positions, window_size, max_comp_128, sparse_mla_topk, ratios):
    device = token_positions.device
    num_tokens = token_positions.shape[0]

    positions = token_positions.unsqueeze(1)
    swa_offsets = torch.arange(window_size, dtype=torch.int32, device=device)
    swa_start = (positions - window_size + 1).clamp(min=0)
    swa_indices = swa_start + swa_offsets
    swa = torch.where(swa_indices > positions, -1, swa_indices).to(torch.int32)

    num_valid = (token_positions + 1) // 128
    comp_col = torch.arange(max_comp_128, dtype=torch.int32, device=device)
    valid_mask = comp_col.unsqueeze(0) < num_valid.unsqueeze(1)
    comp = torch.where(
        valid_mask,
        comp_col.unsqueeze(0).expand(num_tokens, -1),
        torch.full((num_tokens, max_comp_128), -1, dtype=torch.int32, device=device),
    )

    kv_lens = token_positions + 1
    lens = {}
    for r in ratios:
        if r == 1:
            total = kv_lens.clamp(max=window_size)
        elif r == 4:
            total = window_size + (kv_lens // r).clamp(max=sparse_mla_topk)
        elif r == 128:
            total = window_size + kv_lens // r
        else:
            raise ValueError(r)
        lens[r] = total.to(torch.int32)
    return swa, comp, lens


def _ref_per_ratio(kv_lens, cached_tokens, ratios):
    out = {}
    for r in ratios:
        compressed = (kv_lens // r).to(torch.int32)
        past = (cached_tokens // r).to(torch.int32)
        new_comp = compressed - past
        cu = torch.nn.functional.pad(torch.cumsum(new_comp, dim=0), (1, 0)).to(torch.int32)
        out[r] = (compressed, past, new_comp, cu)
    return out


def _ref_mask(new_comp, cu, total_tokens, batch_size, device):
    token_idx = torch.arange(total_tokens, dtype=torch.int32, device=device)
    seq_idx = torch.searchsorted(cu[1:], token_idx, right=True).clamp_(max=batch_size - 1)
    return (token_idx - cu[seq_idx]) < new_comp[seq_idx]


def _ref_ctx_position_ids(past_kv, cu, total, ratio, num_contexts, device):
    ctx_idx = torch.arange(total, dtype=torch.int32, device=device)
    ctx_cu = cu[: num_contexts + 1].to(torch.int32)
    ctx_req = torch.searchsorted(ctx_cu[1:], ctx_idx, right=True)
    return ((past_kv[:num_contexts][ctx_req] + (ctx_idx - ctx_cu[ctx_req])) * ratio).to(torch.int32)


def _ref_gen_position_ids(past_kv, cu, gen_comp, offset, ratio, num_contexts, batch_size, device):
    output_idx = torch.arange(gen_comp, dtype=torch.int32, device=device) + offset
    req_idx = torch.searchsorted(cu[: batch_size + 1], output_idx, right=True) - 1
    req_idx = req_idx.clamp(min=num_contexts, max=batch_size - 1)
    return ((past_kv[req_idx] + (output_idx - cu[req_idx])) * ratio).to(torch.int32)


def _ref_token_positions(seq_lens, cached_tokens, batch_size, num_tokens, device):
    cu = torch.nn.functional.pad(torch.cumsum(seq_lens.to(torch.int), dim=0), (1, 0)).to(
        torch.int32
    )
    token_idx = torch.arange(num_tokens, dtype=torch.int32, device=device)
    req_idx = torch.searchsorted(cu[1 : batch_size + 1].to(torch.int32), token_idx, right=True)
    positions = cached_tokens[req_idx].to(torch.int32) + (token_idx - cu[req_idx].to(torch.int32))
    return cu, req_idx.to(torch.int32), positions


# -------------------------------------------------------------------- tests


@pytest.mark.parametrize("ratios", RATIO_SETS)
@pytest.mark.parametrize(
    "positions_spec",
    [
        "zeros",
        "boundaries",
        "window_edge",
        "topk_saturating",
        "random_long",
        "decode_like",
    ],
)
@pytest.mark.parametrize("window_size,max_comp,topk", [(128, 256, 2048), (128, 8, 64), (64, 4, 16)])
def test_compute_indices(ratios, positions_spec, window_size, max_comp, topk):
    device = "cuda"
    torch.manual_seed(0)
    if positions_spec == "zeros":
        tp = torch.zeros(5, dtype=torch.int32, device=device)
    elif positions_spec == "boundaries":
        tp = torch.tensor([0, 1, 127, 128, 255, 256, 511, 512], dtype=torch.int32, device=device)
    elif positions_spec == "window_edge":
        tp = torch.tensor(
            [window_size - 2, window_size - 1, window_size, window_size + 1],
            dtype=torch.int32,
            device=device,
        )
    elif positions_spec == "topk_saturating":
        tp = torch.tensor(
            [topk * 4 - 1, topk * 4, topk * 4 + 1, 100000], dtype=torch.int32, device=device
        )
    elif positions_spec == "random_long":
        tp = torch.randint(0, 40000, (129,), dtype=torch.int32, device=device)
    else:
        tp = torch.randint(1000, 4000, (256,), dtype=torch.int32, device=device)

    num_tokens = tp.shape[0]
    ref_swa, ref_comp, ref_lens = _ref_indices(tp, window_size, max_comp, topk, ratios)

    # Deliberately over-allocate rows so the op must honour num_tokens, not the
    # buffer shape (mirrors the CUDA-graph padded buffers in production).
    swa = torch.full((num_tokens + 7, window_size), 123, dtype=torch.int32, device=device)
    comp = torch.full((num_tokens + 7, max_comp), 123, dtype=torch.int32, device=device)
    lens = {r: torch.full((num_tokens + 7,), 123, dtype=torch.int32, device=device) for r in ratios}

    torch.ops.trtllm.deepseek_v4_compute_indices(
        tp,
        window_size,
        max_comp,
        topk,
        swa,
        comp,
        lens.get(1),
        lens.get(4),
        lens.get(128),
    )

    torch.testing.assert_close(swa[:num_tokens], ref_swa, rtol=0, atol=0)
    torch.testing.assert_close(comp[:num_tokens], ref_comp, rtol=0, atol=0)
    for r in ratios:
        torch.testing.assert_close(lens[r][:num_tokens], ref_lens[r], rtol=0, atol=0)
    # Padding rows must be untouched.
    assert (swa[num_tokens:] == 123).all()
    assert (comp[num_tokens:] == 123).all()


@pytest.mark.parametrize("ratios", RATIO_SETS)
@pytest.mark.parametrize("batch_size", [1, 2, 7, 37, 64, 146, 257, 300])
def test_compute_per_ratio_kv_lens(ratios, batch_size):
    device = "cuda"
    torch.manual_seed(batch_size)
    kv_lens = torch.randint(1, 40000, (batch_size,), dtype=torch.int32, device=device)
    cached = (kv_lens * torch.rand(batch_size, device=device)).to(torch.int32)
    ref = _ref_per_ratio(kv_lens, cached, ratios)

    pad = 5
    compressed = {
        r: torch.zeros(batch_size + pad, dtype=torch.int32, device=device) for r in ratios
    }
    past = {r: torch.zeros(batch_size + pad, dtype=torch.int32, device=device) for r in ratios}
    new_comp = {r: torch.zeros(batch_size + pad, dtype=torch.int32, device=device) for r in ratios}
    cu = {r: torch.zeros(batch_size + 1 + pad, dtype=torch.int32, device=device) for r in ratios}

    torch.ops.trtllm.deepseek_v4_compute_per_ratio_kv_lens(
        kv_lens,
        cached,
        ratios,
        [compressed[r] for r in ratios],
        [past[r] for r in ratios],
        [new_comp[r] for r in ratios],
        [cu[r] for r in ratios],
    )

    for r in ratios:
        exp_c, exp_p, exp_n, exp_cu = ref[r]
        torch.testing.assert_close(compressed[r][:batch_size], exp_c, rtol=0, atol=0)
        torch.testing.assert_close(past[r][:batch_size], exp_p, rtol=0, atol=0)
        torch.testing.assert_close(new_comp[r][:batch_size], exp_n, rtol=0, atol=0)
        torch.testing.assert_close(cu[r][: batch_size + 1], exp_cu, rtol=0, atol=0)


@pytest.mark.parametrize("ratios", RATIO_SETS)
@pytest.mark.parametrize("batch_size", [1, 3, 37, 146, 257])
def test_compute_compressed_mask(ratios, batch_size):
    device = "cuda"
    torch.manual_seed(batch_size + 11)
    kv_lens = torch.randint(1, 8000, (batch_size,), dtype=torch.int32, device=device)
    cached = (kv_lens * torch.rand(batch_size, device=device)).to(torch.int32)
    ref = _ref_per_ratio(kv_lens, cached, ratios)

    totals = {r: int(ref[r][3][batch_size].item()) for r in ratios}
    masks = {r: torch.zeros(max(totals[r], 1) + 8, dtype=torch.bool, device=device) for r in ratios}
    torch.ops.trtllm.deepseek_v4_compute_compressed_mask(
        [ref[r][2] for r in ratios],
        [ref[r][3] for r in ratios],
        [masks[r] for r in ratios],
        [totals[r] for r in ratios],
        batch_size,
    )
    for r in ratios:
        if totals[r] == 0:
            continue
        expected = _ref_mask(ref[r][2], ref[r][3], totals[r], batch_size, device)
        torch.testing.assert_close(masks[r][: totals[r]], expected, rtol=0, atol=0)


@pytest.mark.parametrize("ratios", RATIO_SETS)
@pytest.mark.parametrize("num_contexts", [1, 4, 33])
def test_compute_ctx_compressed_position_ids(ratios, num_contexts):
    device = "cuda"
    torch.manual_seed(num_contexts + 7)
    kv_lens = torch.randint(1, 6000, (num_contexts,), dtype=torch.int32, device=device)
    cached = (kv_lens * torch.rand(num_contexts, device=device)).to(torch.int32)
    ref = _ref_per_ratio(kv_lens, cached, ratios)

    counts = {r: int(ref[r][3][num_contexts].item()) for r in ratios}
    out = {r: torch.zeros(max(counts[r], 1) + 8, dtype=torch.int32, device=device) for r in ratios}
    torch.ops.trtllm.deepseek_v4_compute_ctx_compressed_position_ids(
        [ref[r][1] for r in ratios],
        [ref[r][3] for r in ratios],
        [out[r] for r in ratios],
        ratios,
        [counts[r] for r in ratios],
        num_contexts,
    )
    for r in ratios:
        if counts[r] == 0:
            continue
        expected = _ref_ctx_position_ids(ref[r][1], ref[r][3], counts[r], r, num_contexts, device)
        torch.testing.assert_close(out[r][: counts[r]], expected, rtol=0, atol=0)


@pytest.mark.parametrize("ratios", RATIO_SETS)
@pytest.mark.parametrize(
    "num_contexts,num_generations,gen_tokens_per_seq",
    [
        (0, 8, 1),
        (0, 16, 4),
        (2, 6, 1),
        (5, 11, 4),
        (0, 1, 1),
    ],
)
def test_compute_gen_compressed_position_ids(
    ratios, num_contexts, num_generations, gen_tokens_per_seq
):
    device = "cuda"
    batch_size = num_contexts + num_generations
    torch.manual_seed(batch_size + gen_tokens_per_seq)
    kv_lens = torch.randint(1, 6000, (batch_size,), dtype=torch.int32, device=device)
    cached = (kv_lens * torch.rand(batch_size, device=device)).to(torch.int32)
    ref = _ref_per_ratio(kv_lens, cached, ratios)

    counts, offsets = {}, {}
    for r in ratios:
        counts[r] = num_generations * ((gen_tokens_per_seq + r - 1) // r)
        offsets[r] = int(ref[r][3][num_contexts].item()) if num_contexts > 0 else 0

    out = {
        r: torch.zeros(offsets[r] + counts[r] + 8, dtype=torch.int32, device=device) for r in ratios
    }
    torch.ops.trtllm.deepseek_v4_compute_gen_compressed_position_ids(
        [ref[r][1] for r in ratios],
        [ref[r][3] for r in ratios],
        [out[r] for r in ratios],
        ratios,
        [counts[r] for r in ratios],
        [offsets[r] for r in ratios],
        num_contexts,
        batch_size,
    )
    for r in ratios:
        if counts[r] == 0:
            continue
        expected = _ref_gen_position_ids(
            ref[r][1], ref[r][3], counts[r], offsets[r], r, num_contexts, batch_size, device
        )
        torch.testing.assert_close(
            out[r][offsets[r] : offsets[r] + counts[r]], expected, rtol=0, atol=0
        )


@pytest.mark.parametrize("seq_lens_spec", ["uniform_1", "mixed", "single", "long_ctx"])
@pytest.mark.parametrize("batch_size", [1, 2, 8, 64, 146, 257])
def test_compute_token_positions(seq_lens_spec, batch_size):
    device = "cuda"
    torch.manual_seed(batch_size + 3)
    if seq_lens_spec == "uniform_1":
        seq_lens = torch.ones(batch_size, dtype=torch.int32, device=device)
    elif seq_lens_spec == "mixed":
        seq_lens = torch.randint(1, 40, (batch_size,), dtype=torch.int32, device=device)
    elif seq_lens_spec == "single":
        seq_lens = torch.full((batch_size,), 7, dtype=torch.int32, device=device)
    else:
        seq_lens = torch.randint(1, 600, (batch_size,), dtype=torch.int32, device=device)

    num_tokens = int(seq_lens.sum().item())
    cached = torch.randint(0, 30000, (batch_size,), dtype=torch.int32, device=device)
    ref_cu, ref_req, ref_pos = _ref_token_positions(
        seq_lens, cached, batch_size, num_tokens, device
    )

    cu = torch.zeros(batch_size + 1 + 4, dtype=torch.int32, device=device)
    req = torch.zeros(num_tokens + 4, dtype=torch.int32, device=device)
    pos = torch.zeros(num_tokens + 4, dtype=torch.int32, device=device)

    torch.ops.trtllm.compute_token_positions(seq_lens, cached, cu, req, pos, num_tokens, True)
    torch.testing.assert_close(cu[: batch_size + 1], ref_cu, rtol=0, atol=0)
    torch.testing.assert_close(req[:num_tokens], ref_req, rtol=0, atol=0)
    torch.testing.assert_close(pos[:num_tokens], ref_pos, rtol=0, atol=0)

    # req_idx must equal the CPU repeat_interleave form the base class used.
    expected_req = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=device), seq_lens
    )
    torch.testing.assert_close(req[:num_tokens], expected_req, rtol=0, atol=0)

    # reuse mode: cu already populated, only the per-token phase runs
    req2 = torch.zeros(num_tokens + 4, dtype=torch.int32, device=device)
    pos2 = torch.zeros(num_tokens + 4, dtype=torch.int32, device=device)
    torch.ops.trtllm.compute_token_positions(seq_lens, cached, cu, req2, pos2, num_tokens, False)
    torch.testing.assert_close(req2[:num_tokens], ref_req, rtol=0, atol=0)
    torch.testing.assert_close(pos2[:num_tokens], ref_pos, rtol=0, atol=0)

    # req-only mode: token_positions omitted
    req3 = torch.zeros(num_tokens + 4, dtype=torch.int32, device=device)
    torch.ops.trtllm.compute_token_positions(seq_lens, None, cu, req3, None, num_tokens, True)
    torch.testing.assert_close(req3[:num_tokens], ref_req, rtol=0, atol=0)


def test_compute_indices_rejects_bad_shapes():
    device = "cuda"
    tp = torch.zeros(4, dtype=torch.int32, device=device)
    swa = torch.zeros((4, 8), dtype=torch.int32, device=device)
    comp = torch.zeros((4, 4), dtype=torch.int32, device=device)
    # window_size larger than the buffer's columns
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.deepseek_v4_compute_indices(tp, 16, 4, 8, swa, comp, None, None, None)
    # wrong dtype
    tp_f = torch.zeros(4, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.deepseek_v4_compute_indices(tp_f, 8, 4, 8, swa, comp, None, None, None)


def test_per_ratio_rejects_mismatched_lists():
    device = "cuda"
    kv = torch.ones(4, dtype=torch.int32, device=device)
    cached = torch.zeros(4, dtype=torch.int32, device=device)
    buf = torch.zeros(4, dtype=torch.int32, device=device)
    cu = torch.zeros(5, dtype=torch.int32, device=device)
    # two ratios but only one buffer per list
    with pytest.raises(RuntimeError):
        torch.ops.trtllm.deepseek_v4_compute_per_ratio_kv_lens(
            kv, cached, [1, 4], [buf], [buf], [buf], [cu]
        )


def _ref_shared_block_table(block_offsets, pool_id, copy_idx, scale, bad_page_index=-1):
    """Reference for ``DeepseekV4CacheManager._compute_shared_block_table``.

    The python path gathered ``block_offsets[pool_id, copy_idx, 0, :]`` on the host
    and mapped it with ``where(base == BAD_PAGE_INDEX, BAD_PAGE_INDEX, base * scale)``.
    """
    base = block_offsets[pool_id, copy_idx, 0, :]
    return torch.where(base == bad_page_index, bad_page_index, base * scale)


@pytest.mark.parametrize("scale", [1, 4, 128])
@pytest.mark.parametrize("num_pools,capacity,max_blocks", [(1, 8, 4), (3, 64, 16), (7, 130, 129)])
@pytest.mark.parametrize("num_seqs", [1, 5, 64])
def test_compute_shared_block_table(scale, num_pools, capacity, max_blocks, num_seqs):
    device = "cuda"
    torch.manual_seed(num_seqs * 31 + max_blocks)
    if num_seqs > capacity:
        pytest.skip("num_seqs must fit the mapper capacity")

    block_offsets = torch.randint(
        0, 1 << 20, (num_pools, capacity, 2, max_blocks), dtype=torch.int32, device=device
    )
    # Sprinkle BAD_PAGE_INDEX so the passthrough branch is exercised, including a
    # fully-invalid row and a fully-valid row.
    block_offsets[block_offsets % 5 == 0] = -1
    block_offsets[:, 0, 0, :] = -1
    if capacity > 1:
        block_offsets[:, 1, 0, :] = 7

    copy_idx = torch.randperm(capacity, device=device)[:num_seqs].to(torch.int32)
    pool_id = num_pools - 1

    out = torch.full((num_seqs, max_blocks), 12345, dtype=torch.int32, device=device)
    torch.ops.trtllm.compute_shared_block_table(block_offsets, copy_idx, pool_id, scale, out)

    expected = _ref_shared_block_table(block_offsets, pool_id, copy_idx.long(), scale)
    torch.testing.assert_close(out, expected.to(torch.int32), rtol=0, atol=0)


def test_compute_shared_block_table_leaves_padding_untouched():
    """Rows past num_seqs must not be written: padded CUDA-graph slots read them."""
    device = "cuda"
    block_offsets = torch.ones((2, 16, 2, 8), dtype=torch.int32, device=device)
    copy_idx = torch.arange(4, dtype=torch.int32, device=device)
    out = torch.full((16, 8), -7, dtype=torch.int32, device=device)

    torch.ops.trtllm.compute_shared_block_table(block_offsets, copy_idx, 1, 4, out[:4])

    torch.testing.assert_close(
        out[4:], torch.full((12, 8), -7, dtype=torch.int32, device=device), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out[:4], torch.full((4, 8), 4, dtype=torch.int32, device=device), rtol=0, atol=0
    )


# The shared-memory prefix scans pick a template instantiation from a size ladder
# (<=512, <=2048, else the compile-time bound). Batch sizes that straddle those
# boundaries take different code paths, and the largest tier is only safe because
# the op layer rejects anything above the bound. Both need coverage.

_SCAN_TIER_BATCHES = [511, 512, 513, 2047, 2048, 2049, 4096]


@pytest.mark.parametrize("batch_size", _SCAN_TIER_BATCHES)
def test_token_positions_across_scan_tiers(batch_size):
    device = "cuda"
    torch.manual_seed(batch_size)
    seq_lens = torch.randint(1, 5, (batch_size,), dtype=torch.int32, device=device)
    num_tokens = int(seq_lens.sum().item())
    cached = torch.randint(0, 1000, (batch_size,), dtype=torch.int32, device=device)

    ref_cu, ref_req, ref_pos = _ref_token_positions(
        seq_lens, cached, batch_size, num_tokens, device
    )
    cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    req = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    pos = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    torch.ops.trtllm.compute_token_positions(seq_lens, cached, cu, req, pos, num_tokens, True)
    torch.testing.assert_close(cu, ref_cu, rtol=0, atol=0)
    torch.testing.assert_close(req, ref_req, rtol=0, atol=0)
    torch.testing.assert_close(pos, ref_pos, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", _SCAN_TIER_BATCHES)
def test_per_ratio_kv_lens_across_scan_tiers(batch_size):
    device = "cuda"
    torch.manual_seed(batch_size + 7)
    ratios = [1, 4, 128]
    kv_lens = torch.randint(1, 9000, (batch_size,), dtype=torch.int32, device=device)
    cached = (kv_lens * torch.rand(batch_size, device=device)).to(torch.int32)
    ref = _ref_per_ratio(kv_lens, cached, ratios)

    comp = {r: torch.zeros(batch_size, dtype=torch.int32, device=device) for r in ratios}
    past = {r: torch.zeros(batch_size, dtype=torch.int32, device=device) for r in ratios}
    new_comp = {r: torch.zeros(batch_size, dtype=torch.int32, device=device) for r in ratios}
    cu = {r: torch.zeros(batch_size + 1, dtype=torch.int32, device=device) for r in ratios}

    torch.ops.trtllm.deepseek_v4_compute_per_ratio_kv_lens(
        kv_lens,
        cached,
        ratios,
        [comp[r] for r in ratios],
        [past[r] for r in ratios],
        [new_comp[r] for r in ratios],
        [cu[r] for r in ratios],
    )
    for r in ratios:
        exp_c, exp_p, exp_n, exp_cu = ref[r]
        torch.testing.assert_close(comp[r], exp_c, rtol=0, atol=0)
        torch.testing.assert_close(past[r], exp_p, rtol=0, atol=0)
        torch.testing.assert_close(new_comp[r], exp_n, rtol=0, atol=0)
        torch.testing.assert_close(cu[r], exp_cu, rtol=0, atol=0)


def test_token_positions_rejects_batch_above_scan_bound():
    """Above the compile-time bound the op must fail loudly, not scribble."""
    device = "cuda"
    too_big = 4097
    seq_lens = torch.ones(too_big, dtype=torch.int32, device=device)
    cached = torch.zeros(too_big, dtype=torch.int32, device=device)
    cu = torch.zeros(too_big + 1, dtype=torch.int32, device=device)
    req = torch.zeros(too_big, dtype=torch.int32, device=device)
    pos = torch.zeros(too_big, dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError):
        torch.ops.trtllm.compute_token_positions(seq_lens, cached, cu, req, pos, too_big, True)

    # With the scan skipped (cu_seq_lens supplied by the caller) the bound does
    # not apply, so the same batch must go through.
    cu_ref = torch.zeros(too_big + 1, dtype=torch.int32, device=device)
    cu_ref[1:] = torch.cumsum(seq_lens, 0, dtype=torch.int32)
    torch.ops.trtllm.compute_token_positions(seq_lens, cached, cu_ref, req, pos, too_big, False)
    expected_req = torch.arange(too_big, dtype=torch.int32, device=device)
    torch.testing.assert_close(req, expected_req, rtol=0, atol=0)


def test_per_ratio_kv_lens_rejects_batch_above_scan_bound():
    device = "cuda"
    too_big = 4097
    kv = torch.ones(too_big, dtype=torch.int32, device=device)
    cached = torch.zeros(too_big, dtype=torch.int32, device=device)
    buf = torch.zeros(too_big, dtype=torch.int32, device=device)
    cu = torch.zeros(too_big + 1, dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError):
        torch.ops.trtllm.deepseek_v4_compute_per_ratio_kv_lens(
            kv, cached, [1], [buf], [buf], [buf], [cu]
        )
