# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the TRT-LLM attention backend custom op.

Mirrors the structure of test_flashinfer_attention_op.py but exercises the
``trtllm_attention_mha_with_cache`` custom op via ``thop.attention``.
"""

import math

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.attention.trtllm_attention import (
    _GlobalTrtllmPlanner,
    prepare_trtllm_metadata_host,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_trtllm_planner():
    """Force a full reset of the global TRT-LLM planner so buffers are re-allocated."""
    _GlobalTrtllmPlanner.__init__()


def _prepare_and_run(
    q,
    k,
    v,
    kv_cache,
    seq_lens,
    input_positions,
    cache_locs,
    pages_per_seq,
    max_seq_len,
    max_batch_size,
    num_prefill,
    num_prefill_tokens,
    num_decode,
    device,
    scale=None,
):
    """Build metadata, call host-prepare, and invoke the TRT-LLM attention op."""
    tokens_per_block = kv_cache.shape[3]
    max_blocks_per_seq = math.ceil(max_seq_len / tokens_per_block)
    block_offset_multiplier = kv_cache.stride(0) // kv_cache.stride(1)

    seq_len_with_cache = [ip + sl for ip, sl in zip(input_positions, seq_lens)]

    # --- tensors for the op ---------------------------------------------------
    batch_info_host = torch.tensor(
        [num_prefill, num_prefill_tokens, num_decode], dtype=torch.int32, device=device
    )
    seq_len_d = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    seq_len_h = torch.tensor(seq_lens, dtype=torch.int32).pin_memory()
    input_pos_h = torch.tensor(input_positions, dtype=torch.int32).pin_memory()
    slwc_d = torch.tensor(seq_len_with_cache, dtype=torch.int32, device=device)
    slwc_h = torch.tensor(seq_len_with_cache, dtype=torch.int32).pin_memory()
    max_seq_info_h = torch.tensor(
        [max_seq_len, max_blocks_per_seq, block_offset_multiplier, max_batch_size],
        dtype=torch.int32,
    ).pin_memory()

    # --- paging metadata -------------------------------------------------------
    cache_loc_d = torch.tensor(cache_locs, dtype=torch.int32, device=device)

    cu_num_pages = [0]
    for pps in pages_per_seq:
        cu_num_pages.append(cu_num_pages[-1] + pps)
    cu_num_pages_d = torch.tensor(cu_num_pages, dtype=torch.int32, device=device)

    # --- host prepare (pinned host tensors) ------------------------------------
    prepare_trtllm_metadata_host(
        batch_info_host=batch_info_host.cpu(),
        max_seq_info_host=max_seq_info_h,
        seq_len_with_cache_host=slwc_h,
        input_pos_host=input_pos_h,
        seq_len_host=seq_len_h,
    )

    # --- device prepare (block_offsets via Triton kernel) ---------------------
    (block_offsets,) = torch.ops.auto_deploy.trtllm_attention_prepare_metadata(
        batch_info_host, max_seq_info_h, cu_num_pages_d, cache_loc_d
    )

    # --- call op --------------------------------------------------------------
    return torch.ops.auto_deploy.trtllm_attention_mha_with_cache(
        q,
        k,
        v,
        batch_info_host,
        seq_len_d,
        slwc_d,
        max_seq_info_h,
        block_offsets,
        kv_cache,
        scale,
    )


# ---------------------------------------------------------------------------
# Test 1 – Context (prefill)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_length", [8, 32, 2048])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_attention_op_context(seq_length, n_heads, batch_size, dtype, device):
    D_HEAD = 64
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 32

    _reset_trtllm_planner()

    # Q, K, V in bsnd layout
    q = torch.randn(batch_size, seq_length, n_heads, D_HEAD, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_length, n_heads, D_HEAD, dtype=dtype, device=device)
    v = torch.randn(batch_size, seq_length, n_heads, D_HEAD, dtype=dtype, device=device)

    # KV cache – HND: [num_blocks, 2, num_heads, tokens_per_block, head_dim]
    # Unpaged: 1 block per sequence, tokens_per_block = MAX_SEQ_LEN
    kv_cache = torch.zeros(
        MAX_BATCH_SIZE, 2, n_heads, MAX_SEQ_LEN, D_HEAD, dtype=dtype, device=device
    )

    output = _prepare_and_run(
        q,
        k,
        v,
        kv_cache,
        seq_lens=[seq_length] * batch_size,
        input_positions=[0] * batch_size,
        cache_locs=list(range(batch_size)),
        pages_per_seq=[1] * batch_size,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        num_prefill=batch_size,
        num_prefill_tokens=batch_size * seq_length,
        num_decode=0,
        device=device,
    )

    # Reference: SDPA causal
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(1, 2)

    assert torch.allclose(
        output.cpu().to(torch.float32), ref.cpu().to(torch.float32), atol=1e-2, rtol=1e-2
    )


# ---------------------------------------------------------------------------
# Test 2 – Decode (generate) with pre-filled cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefill_seq_length", [1, 4, 2047])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_attention_op_decode(prefill_seq_length, batch_size, n_heads, dtype, device):
    D_HEAD = 64
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 32

    _reset_trtllm_planner()

    # --- Step 1: prefill to populate the cache --------------------------------
    q_pf = torch.randn(batch_size, prefill_seq_length, n_heads, D_HEAD, dtype=dtype, device=device)
    k_pf = torch.randn(batch_size, prefill_seq_length, n_heads, D_HEAD, dtype=dtype, device=device)
    v_pf = torch.randn(batch_size, prefill_seq_length, n_heads, D_HEAD, dtype=dtype, device=device)

    kv_cache = torch.zeros(
        MAX_BATCH_SIZE, 2, n_heads, MAX_SEQ_LEN, D_HEAD, dtype=dtype, device=device
    )

    _prepare_and_run(
        q_pf,
        k_pf,
        v_pf,
        kv_cache,
        seq_lens=[prefill_seq_length] * batch_size,
        input_positions=[0] * batch_size,
        cache_locs=list(range(batch_size)),
        pages_per_seq=[1] * batch_size,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        num_prefill=batch_size,
        num_prefill_tokens=batch_size * prefill_seq_length,
        num_decode=0,
        device=device,
    )

    # --- Step 2: decode one token ---------------------------------------------
    q_dec = torch.randn(batch_size, 1, n_heads, D_HEAD, dtype=dtype, device=device)
    k_dec = torch.randn(batch_size, 1, n_heads, D_HEAD, dtype=dtype, device=device)
    v_dec = torch.randn(batch_size, 1, n_heads, D_HEAD, dtype=dtype, device=device)

    _reset_trtllm_planner()

    output = _prepare_and_run(
        q_dec,
        k_dec,
        v_dec,
        kv_cache,
        seq_lens=[1] * batch_size,
        input_positions=[prefill_seq_length] * batch_size,
        cache_locs=list(range(batch_size)),
        pages_per_seq=[1] * batch_size,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        num_prefill=0,
        num_prefill_tokens=0,
        num_decode=batch_size,
        device=device,
    )

    # Reference: Q_dec attends to full K, V (prefill + new)
    k_full = torch.cat([k_pf, k_dec], dim=1)
    v_full = torch.cat([v_pf, v_dec], dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q_dec.transpose(1, 2),
        k_full.transpose(1, 2),
        v_full.transpose(1, 2),
        is_causal=False,
    ).transpose(1, 2)

    assert torch.allclose(
        output.cpu().to(torch.float32), ref.cpu().to(torch.float32), atol=1e-2, rtol=1e-2
    )


# ---------------------------------------------------------------------------
# Test 3 – Context then Generate (full cycle, verify both outputs)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefill_seq_length", [4, 10, 2047])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_attention_context_and_generate(
    prefill_seq_length, n_heads, batch_size, dtype, device
):
    D_HEAD = 64
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 32

    _reset_trtllm_planner()

    # --- Prefill --------------------------------------------------------------
    q_pf = torch.randn(batch_size, prefill_seq_length, n_heads, D_HEAD, dtype=dtype, device=device)
    k_pf = torch.randn(batch_size, prefill_seq_length, n_heads, D_HEAD, dtype=dtype, device=device)
    v_pf = torch.randn(batch_size, prefill_seq_length, n_heads, D_HEAD, dtype=dtype, device=device)

    kv_cache = torch.zeros(
        MAX_BATCH_SIZE, 2, n_heads, MAX_SEQ_LEN, D_HEAD, dtype=dtype, device=device
    )

    output_pf = _prepare_and_run(
        q_pf,
        k_pf,
        v_pf,
        kv_cache,
        seq_lens=[prefill_seq_length] * batch_size,
        input_positions=[0] * batch_size,
        cache_locs=list(range(batch_size)),
        pages_per_seq=[1] * batch_size,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        num_prefill=batch_size,
        num_prefill_tokens=batch_size * prefill_seq_length,
        num_decode=0,
        device=device,
    )

    # Verify prefill output
    ref_pf = torch.nn.functional.scaled_dot_product_attention(
        q_pf.transpose(1, 2), k_pf.transpose(1, 2), v_pf.transpose(1, 2), is_causal=True
    ).transpose(1, 2)

    assert torch.allclose(
        output_pf.cpu().to(torch.float32), ref_pf.cpu().to(torch.float32), atol=1e-2, rtol=1e-2
    )

    # --- Generate one token ---------------------------------------------------
    q_gen = torch.randn(batch_size, 1, n_heads, D_HEAD, dtype=dtype, device=device)
    k_gen = torch.randn(batch_size, 1, n_heads, D_HEAD, dtype=dtype, device=device)
    v_gen = torch.randn(batch_size, 1, n_heads, D_HEAD, dtype=dtype, device=device)

    _reset_trtllm_planner()

    output_gen = _prepare_and_run(
        q_gen,
        k_gen,
        v_gen,
        kv_cache,
        seq_lens=[1] * batch_size,
        input_positions=[prefill_seq_length] * batch_size,
        cache_locs=list(range(batch_size)),
        pages_per_seq=[1] * batch_size,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        num_prefill=0,
        num_prefill_tokens=0,
        num_decode=batch_size,
        device=device,
    )

    # Verify decode output – Q_gen attends to full (prefill + gen) K, V
    k_full = torch.cat([k_pf, k_gen], dim=1)
    v_full = torch.cat([v_pf, v_gen], dim=1)
    ref_gen = torch.nn.functional.scaled_dot_product_attention(
        q_gen.transpose(1, 2),
        k_full.transpose(1, 2),
        v_full.transpose(1, 2),
        is_causal=False,
    ).transpose(1, 2)

    assert torch.allclose(
        output_gen.cpu().to(torch.float32), ref_gen.cpu().to(torch.float32), atol=1e-2, rtol=1e-2
    )


# ---------------------------------------------------------------------------
# Test 4 – Context with non-zero input_pos (chunked prefill)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq",
    [
        (2, 1),
        (2, 64),
        (2, 2046),
        (16, 1),
        (16, 2022),
        (1984, 64),
        (1024, 1024),
    ],
)
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_attention_op_context_input_pos(seq, batch_size, n_heads, dtype, device):
    D_HEAD = 64
    MAX_SEQ_LEN = 2048
    MAX_BATCH_SIZE = 32
    SEQ_LEN = seq[0]
    PREFILL_SEQ_LEN = seq[1]

    _reset_trtllm_planner()

    # --- Step 1: prefill the first chunk to populate cache --------------------
    q_1 = torch.randn(batch_size, PREFILL_SEQ_LEN, n_heads, D_HEAD, dtype=dtype, device=device)
    k_1 = torch.randn(batch_size, PREFILL_SEQ_LEN, n_heads, D_HEAD, dtype=dtype, device=device)
    v_1 = torch.randn(batch_size, PREFILL_SEQ_LEN, n_heads, D_HEAD, dtype=dtype, device=device)

    kv_cache = torch.zeros(
        MAX_BATCH_SIZE, 2, n_heads, MAX_SEQ_LEN, D_HEAD, dtype=dtype, device=device
    )

    _prepare_and_run(
        q_1,
        k_1,
        v_1,
        kv_cache,
        seq_lens=[PREFILL_SEQ_LEN] * batch_size,
        input_positions=[0] * batch_size,
        cache_locs=list(range(batch_size)),
        pages_per_seq=[1] * batch_size,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        num_prefill=batch_size,
        num_prefill_tokens=batch_size * PREFILL_SEQ_LEN,
        num_decode=0,
        device=device,
    )

    # --- Step 2: second context chunk at non-zero input_pos -------------------
    q_2 = torch.randn(batch_size, SEQ_LEN, n_heads, D_HEAD, dtype=dtype, device=device)
    k_2 = torch.randn(batch_size, SEQ_LEN, n_heads, D_HEAD, dtype=dtype, device=device)
    v_2 = torch.randn(batch_size, SEQ_LEN, n_heads, D_HEAD, dtype=dtype, device=device)

    _reset_trtllm_planner()

    output = _prepare_and_run(
        q_2,
        k_2,
        v_2,
        kv_cache,
        seq_lens=[SEQ_LEN] * batch_size,
        input_positions=[PREFILL_SEQ_LEN] * batch_size,
        cache_locs=list(range(batch_size)),
        pages_per_seq=[1] * batch_size,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        num_prefill=batch_size,
        num_prefill_tokens=batch_size * SEQ_LEN,
        num_decode=0,
        device=device,
    )

    # https://nvbugspro.nvidia.com/bug/5904035 "chunked prefill" (context tokens with non-zero `input_pos`) can behave
    # differently depending on which TRT-LLM kernel path is selected.
    #
    # - On some paths (notably SM100 fallback), the context-stage output corresponds to
    #   causal attention over the *current chunk only* (cached prefix is not attended),
    #   though the KV cache is still updated.
    # - On others (e.g. SM80/A30), the output incorporates the cached prefix and matches
    #   full causal attention over (prefix + current chunk) with the appropriate offset.
    #
    # We accept either behavior here, but still require the output to match one of the
    # two well-defined PyTorch SDPA references.
    ref_chunk_only = torch.nn.functional.scaled_dot_product_attention(
        q_2.transpose(1, 2),
        k_2.transpose(1, 2),
        v_2.transpose(1, 2),
        is_causal=True,
    ).transpose(1, 2)

    out_f32 = output.cpu().to(torch.float32)
    if torch.allclose(out_f32, ref_chunk_only.cpu().to(torch.float32), atol=1e-2, rtol=1e-2):
        return

    # Full reference: Q_2 attends to full K (chunk1 + chunk2) with causal mask offset.
    k_full = torch.cat([k_1, k_2], dim=1)
    v_full = torch.cat([v_1, v_2], dim=1)
    mask = torch.cat(
        [
            torch.ones(SEQ_LEN, PREFILL_SEQ_LEN, device=device, dtype=torch.bool),
            torch.tril(torch.ones(SEQ_LEN, SEQ_LEN, device=device, dtype=torch.bool)),
        ],
        dim=1,
    )
    ref_with_prefix = torch.nn.functional.scaled_dot_product_attention(
        q_2.transpose(1, 2),
        k_full.transpose(1, 2),
        v_full.transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)

    assert torch.allclose(out_f32, ref_with_prefix.cpu().to(torch.float32), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 5 – Paged KV cache (prefill + decode with small page_size)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seq_lengths", [[8, 14], [11, 19, 22, 49]])
@pytest.mark.parametrize("n_heads", [8])
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_trtllm_attention_with_paged_kvcache(seq_lengths, n_heads, dtype, device):
    PAGE_SIZE = 8
    D_HEAD = 64
    MAX_SEQ_LEN = 128
    MAX_BATCH_SIZE = 32
    BATCH_SIZE = len(seq_lengths)
    TOTAL_SEQ_LEN = sum(seq_lengths)

    MAX_NUM_PAGES = MAX_BATCH_SIZE * MAX_SEQ_LEN // PAGE_SIZE

    _reset_trtllm_planner()

    # Q, K, V – flattened across batch: [1, total_tokens, n_heads, d_head]
    q = torch.randn(1, TOTAL_SEQ_LEN, n_heads, D_HEAD, dtype=dtype, device=device)
    k = torch.randn(1, TOTAL_SEQ_LEN, n_heads, D_HEAD, dtype=dtype, device=device)
    v = torch.randn(1, TOTAL_SEQ_LEN, n_heads, D_HEAD, dtype=dtype, device=device)

    # Paged KV cache – HND: [num_pages, 2, n_heads, page_size, d_head]
    kv_cache = torch.zeros(MAX_NUM_PAGES, 2, n_heads, PAGE_SIZE, D_HEAD, dtype=dtype, device=device)

    # Assign pages randomly
    free_pages = torch.randperm(MAX_NUM_PAGES).int().tolist()
    pages_per_seq_list = [math.ceil(s / PAGE_SIZE) for s in seq_lengths]
    page_assignments = [[free_pages.pop() for _ in range(np)] for np in pages_per_seq_list]
    cache_locs = [p for ps in page_assignments for p in ps]

    # Prefill
    output_pf = _prepare_and_run(
        q,
        k,
        v,
        kv_cache,
        seq_lens=seq_lengths,
        input_positions=[0] * BATCH_SIZE,
        cache_locs=cache_locs,
        pages_per_seq=pages_per_seq_list,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        num_prefill=BATCH_SIZE,
        num_prefill_tokens=TOTAL_SEQ_LEN,
        num_decode=0,
        device=device,
    )

    # Reference – per-sequence causal SDPA
    cu = [0] + list(torch.cumsum(torch.tensor(seq_lengths), 0).tolist())
    ref_parts = []
    for i, s in enumerate(seq_lengths):
        qq = q[0, cu[i] : cu[i + 1], :, :].unsqueeze(0)
        kk = k[0, cu[i] : cu[i + 1], :, :].unsqueeze(0)
        vv = v[0, cu[i] : cu[i + 1], :, :].unsqueeze(0)
        oo = torch.nn.functional.scaled_dot_product_attention(
            qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
        ref_parts.append(oo.squeeze(0))
    ref = torch.cat(ref_parts, dim=0)

    assert torch.allclose(
        output_pf.squeeze(0).cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )

    # --- Now generate one token per sequence ----------------------------------
    _reset_trtllm_planner()

    q_gen = torch.randn(BATCH_SIZE, 1, n_heads, D_HEAD, dtype=dtype, device=device)
    k_gen = torch.randn(BATCH_SIZE, 1, n_heads, D_HEAD, dtype=dtype, device=device)
    v_gen = torch.randn(BATCH_SIZE, 1, n_heads, D_HEAD, dtype=dtype, device=device)

    # Update page assignments – may need a new page for sequences whose last page is full
    for i, (pages, s) in enumerate(zip(page_assignments, seq_lengths)):
        last_page_occupancy = s % PAGE_SIZE
        if last_page_occupancy == 0:  # last page was full → need a new page
            pages.append(free_pages.pop())
            pages_per_seq_list[i] = len(pages)
    cache_locs_gen = [p for ps in page_assignments for p in ps]

    output_gen = _prepare_and_run(
        q_gen,
        k_gen,
        v_gen,
        kv_cache,
        seq_lens=[1] * BATCH_SIZE,
        input_positions=seq_lengths,
        cache_locs=cache_locs_gen,
        pages_per_seq=pages_per_seq_list,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        num_prefill=0,
        num_prefill_tokens=0,
        num_decode=BATCH_SIZE,
        device=device,
    )

    # Reference for decode
    ref_gen_parts = []
    for i, s in enumerate(seq_lengths):
        qq = q_gen[i : i + 1, :, :, :]  # [1, 1, n_heads, d_head]
        kk = k[0, cu[i] : cu[i + 1], :, :].unsqueeze(0)  # [1, s, n_heads, d_head]
        kk = torch.cat([kk, k_gen[i : i + 1, :, :, :]], dim=1)  # [1, s+1, ...]
        vv = v[0, cu[i] : cu[i + 1], :, :].unsqueeze(0)
        vv = torch.cat([vv, v_gen[i : i + 1, :, :, :]], dim=1)
        oo = torch.nn.functional.scaled_dot_product_attention(
            qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2), is_causal=False
        ).transpose(1, 2)
        ref_gen_parts.append(oo.squeeze(0))
    ref_gen = torch.cat(ref_gen_parts, dim=0)

    assert torch.allclose(
        output_gen.reshape(-1, n_heads, D_HEAD).cpu().to(torch.float32),
        ref_gen.cpu().to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


# ---------------------------------------------------------------------------
# Block offsets computation test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_sequences, pages_per_seq_list, block_offset_multiplier",
    [
        (1, [1], 2),
        (3, [2, 3, 1], 2),
        (4, [1, 4, 2, 3], 2),
        (2, [5, 5], 4),
    ],
)
def test_block_offsets_computation(num_sequences, pages_per_seq_list, block_offset_multiplier):
    """Verify block_offsets produced by prepare_trtllm_metadata device-side op."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    device = "cuda"
    _reset_trtllm_planner()

    max_batch_size = max(num_sequences, 4)
    total_pages = sum(pages_per_seq_list)
    max_blocks_per_seq = max(pages_per_seq_list)
    max_seq_len = max_blocks_per_seq * 64

    cache_locs = list(range(100, 100 + total_pages))
    cache_loc_d = torch.tensor(cache_locs, dtype=torch.int32, device=device)

    cu_num_pages = [0]
    for pps in pages_per_seq_list:
        cu_num_pages.append(cu_num_pages[-1] + pps)
    cu_num_pages_d = torch.tensor(cu_num_pages, dtype=torch.int32, device=device)

    batch_info_host = torch.tensor(
        [num_sequences, num_sequences, 0], dtype=torch.int32
    ).pin_memory()
    max_seq_info_h = torch.tensor(
        [max_seq_len, max_blocks_per_seq, block_offset_multiplier, max_batch_size],
        dtype=torch.int32,
    ).pin_memory()

    prepare_trtllm_metadata_host(
        batch_info_host=batch_info_host,
        max_seq_info_host=max_seq_info_h,
        seq_len_with_cache_host=torch.ones(num_sequences, dtype=torch.int32).pin_memory(),
        input_pos_host=torch.zeros(num_sequences, dtype=torch.int32).pin_memory(),
        seq_len_host=torch.ones(num_sequences, dtype=torch.int32).pin_memory(),
    )

    torch.ops.auto_deploy.trtllm_attention_prepare_metadata(
        batch_info_host, max_seq_info_h, cu_num_pages_d, cache_loc_d
    )

    block_offsets = _GlobalTrtllmPlanner.block_offsets
    bo = block_offsets[0].cpu()

    # Python reference: for each sequence and page, verify K and V offsets
    offset = 0
    for seq_id, n_pages in enumerate(pages_per_seq_list):
        for pg in range(n_pages):
            raw = cache_locs[offset]
            expected_k = raw * block_offset_multiplier
            expected_v = expected_k + 1
            assert bo[seq_id, 0, pg].item() == expected_k, (
                f"K mismatch at seq={seq_id} pg={pg}: "
                f"got {bo[seq_id, 0, pg].item()}, expected {expected_k}"
            )
            assert bo[seq_id, 1, pg].item() == expected_v, (
                f"V mismatch at seq={seq_id} pg={pg}: "
                f"got {bo[seq_id, 1, pg].item()}, expected {expected_v}"
            )
            offset += 1

    # Verify zero-padding in unused slots
    for seq_id, n_pages in enumerate(pages_per_seq_list):
        for pg in range(n_pages, max_blocks_per_seq):
            assert bo[seq_id, 0, pg].item() == 0, f"K padding non-zero at seq={seq_id} pg={pg}"
            assert bo[seq_id, 1, pg].item() == 1, f"V padding not 1 at seq={seq_id} pg={pg}"
