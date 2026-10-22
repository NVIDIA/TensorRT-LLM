# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the opt-in beam_shared_prefix FMHA library.

The merge math is covered on the CPU against a dense reference; the
end-to-end test runs Qwen3-0.6B beam search with and without the library and
requires a GPU plus the model checkpoint.
"""

import math
import os

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention.backends.fmha.beam_shared_prefix import (
    LIB_NAME,
    merge_prefix_and_tail,
)

_LOG2E = 1.4426950408889634


def _reference_prefix_state(
    q: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
):
    """Dense prefix attention plus its base-2 LSE, FlashInfer's convention."""
    rows, num_heads, head_dim = q.shape
    group = num_heads // num_kv_heads
    q_grouped = q.view(rows, num_kv_heads, group, head_dim).float()
    scores = torch.einsum("rkgd,rtkd->rkgt", q_grouped, k_prefix.float())
    scores = scores * sm_scale
    out = torch.einsum("rkgt,rtkd->rkgd", torch.softmax(scores, dim=-1), v_prefix.float())
    lse2 = torch.logsumexp(scores, dim=-1) * _LOG2E
    return out.view(rows, num_heads, head_dim), lse2.view(rows, num_heads)


@pytest.mark.parametrize("tail_len, with_mask", [(1, False), (3, False), (4, True)])
@pytest.mark.parametrize("group", [1, 2, 4])
def test_merge_prefix_and_tail_matches_dense_attention(tail_len, with_mask, group):
    gen = torch.Generator().manual_seed(1234)
    rows, num_kv_heads, head_dim, prefix_len = 6, 2, 64, 37
    num_heads = num_kv_heads * group
    sm_scale = 1.0 / math.sqrt(head_dim)

    q = torch.randn(rows, num_heads, head_dim, generator=gen)
    k_prefix = torch.randn(rows, prefix_len, num_kv_heads, head_dim, generator=gen)
    v_prefix = torch.randn(rows, prefix_len, num_kv_heads, head_dim, generator=gen)
    k_tail = torch.randn(rows, tail_len, num_kv_heads, head_dim, generator=gen)
    v_tail = torch.randn(rows, tail_len, num_kv_heads, head_dim, generator=gen)
    tail_valid = None
    if with_mask:
        # Ragged tails: row r keeps its last (r % tail_len) + 1 positions.
        keep = torch.tensor([(r % tail_len) + 1 for r in range(rows)])
        tail_valid = torch.arange(tail_len).view(1, -1) >= (tail_len - keep).view(-1, 1)

    prefix_out, prefix_lse = _reference_prefix_state(q, k_prefix, v_prefix, num_kv_heads, sm_scale)
    merged = merge_prefix_and_tail(
        q,
        k_tail,
        v_tail,
        tail_valid,
        prefix_out,
        prefix_lse,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
    )

    # Dense reference over the concatenated prefix + (valid) tail.
    k_all = torch.cat([k_prefix, k_tail], dim=1)
    v_all = torch.cat([v_prefix, v_tail], dim=1)
    q_grouped = q.view(rows, num_kv_heads, group, head_dim)
    scores = torch.einsum("rkgd,rtkd->rkgt", q_grouped, k_all) * sm_scale
    if tail_valid is not None:
        valid = torch.cat([torch.ones(rows, prefix_len, dtype=torch.bool), tail_valid], dim=1)
        scores = scores.masked_fill(~valid.view(rows, 1, 1, -1), float("-inf"))
    expected = torch.einsum("rkgt,rtkd->rkgd", torch.softmax(scores, dim=-1), v_all).view(
        rows, num_heads, head_dim
    )
    torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-4)


def _qwen3_small_path():
    root = llm_models_root(check=False)
    if root is None:
        return None
    path = root / "Qwen3" / "Qwen3-0.6B"
    return path if path.exists() else None


def _run_beam_search(model_path, prompts, beam_width, max_tokens, env):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        llm = LLM(
            model=str(model_path),
            max_beam_width=beam_width,
            max_batch_size=len(prompts),
            max_num_tokens=max(len(p) for p in prompts) * len(prompts),
            max_seq_len=max(len(p) for p in prompts) + max_tokens + 1,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3),
            cuda_graph_config=CudaGraphConfig(batch_sizes=[len(prompts)]),
            skip_tokenizer_init=True,
        )
        with llm:
            sampling = SamplingParams(
                use_beam_search=True,
                n=beam_width,
                max_tokens=max_tokens,
                ignore_eos=True,
                end_id=-1,
                logprobs=1,
            )
            outputs = llm.generate(prompts, sampling_params=sampling)
        return [
            [(tuple(beam.token_ids), beam.cumulative_logprob) for beam in out.outputs]
            for out in outputs
        ]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(_qwen3_small_path() is None, reason="Qwen3-0.6B checkpoint not available")
@pytest.mark.threadleak(enabled=False)
def test_beam_shared_prefix_matches_fallback_e2e():
    """Beam search with the library reproduces the fallback kernel's beams.

    Three output tokens over a prompt that spans several KV blocks, two
    requests in one batch, CUDA graphs on: the second and third decode steps
    exercise the indirection-gathered tail, the first the empty tail.
    """
    model_path = _qwen3_small_path()
    vocab = 150000
    prompts = [
        [1024 + ((17 * i + 13 * p) % (vocab - 1024)) for p in range(n)]
        for i, n in enumerate((300, 173))
    ]
    beam_width, max_tokens = 16, 3

    base_env = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}
    reference = _run_beam_search(model_path, prompts, beam_width, max_tokens, base_env)
    # Check mode synchronizes after every stage and validates the page table
    # and the prefix attention against dense references inside the library.
    candidate = _run_beam_search(
        model_path,
        prompts,
        beam_width,
        max_tokens,
        base_env
        | {
            "TLLM_FMHA_LIBS": f"+{LIB_NAME}",
            "TLLM_BEAM_SHARED_PREFIX_CHECK": "1",
            "TLLM_BEAM_SHARED_PREFIX_MAX_TAIL": str(max_tokens - 1),
        },
    )

    for ref_beams, cand_beams in zip(reference, candidate):
        assert len(cand_beams) == beam_width
        ref_tokens = [tokens for tokens, _ in ref_beams]
        cand_tokens = [tokens for tokens, _ in cand_beams]
        # Kernel numerics differ slightly, so allow the ranking of a few
        # near-tied beams to move; the sets must agree almost entirely.
        overlap = len(set(ref_tokens) & set(cand_tokens)) / beam_width
        assert overlap >= 0.9, (overlap, ref_tokens, cand_tokens)
        assert cand_tokens[0] == ref_tokens[0]
        for (tokens, ref_lp), (_, cand_lp) in zip(ref_beams, cand_beams):
            if ref_lp is not None and cand_lp is not None:
                assert abs(ref_lp - cand_lp) < 0.1 + 0.02 * abs(ref_lp)


# ---------------------------------------------------------------------------
# n-sample (beam_width == 1) path
# ---------------------------------------------------------------------------
def test_merge_states_matches_dense_attention():
    from tensorrt_llm._torch.attention.backends.fmha.beam_shared_prefix import merge_states

    gen = torch.Generator().manual_seed(7)
    rows, num_kv_heads, group, head_dim = 5, 2, 2, 64
    num_heads = num_kv_heads * group
    sm_scale = 1.0 / math.sqrt(head_dim)
    q = torch.randn(rows, num_heads, head_dim, generator=gen)
    k_a = torch.randn(rows, 40, num_kv_heads, head_dim, generator=gen)
    v_a = torch.randn(rows, 40, num_kv_heads, head_dim, generator=gen)
    k_b = torch.randn(rows, 9, num_kv_heads, head_dim, generator=gen)
    v_b = torch.randn(rows, 9, num_kv_heads, head_dim, generator=gen)

    out_a, lse_a = _reference_prefix_state(q, k_a, v_a, num_kv_heads, sm_scale)
    out_b, lse_b = _reference_prefix_state(q, k_b, v_b, num_kv_heads, sm_scale)
    a_empty = torch.tensor([False, True, False, False, True])
    # A row flagged empty takes state b alone, even if state a is garbage.
    out_a[a_empty] = float("nan")
    merged = merge_states(out_a, lse_a, out_b, lse_b, a_empty)

    expected, _ = _reference_prefix_state(
        q, torch.cat([k_a, k_b], dim=1), torch.cat([v_a, v_b], dim=1), num_kv_heads, sm_scale
    )
    expected_b, _ = _reference_prefix_state(q, k_b, v_b, num_kv_heads, sm_scale)
    expected[a_empty] = expected_b[a_empty]
    torch.testing.assert_close(merged, expected, atol=1e-5, rtol=1e-4)


def _fake_sampling_metadata(block_tables, kv_lens, tokens_per_block, num_pages):
    """Minimal metadata for the host planner: one pool, page stride 2."""
    from types import SimpleNamespace

    rows = len(block_tables)
    max_blocks = max(len(t) for t in block_tables)
    offsets = torch.zeros(1, rows, 2, max_blocks, dtype=torch.int32)
    for r, table in enumerate(block_tables):
        offsets[0, r, 0, : len(table)] = torch.tensor(table) * 2
    manager = SimpleNamespace(
        layer_offsets={0: 0},
        kv_factor=2,
        get_buffers=lambda layer, kv_layout: torch.empty(num_pages, 1),
    )
    return SimpleNamespace(
        fmha_plan_caches={},
        kv_cache_manager=manager,
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]]),
        host_kv_cache_block_offsets=offsets,
        kv_lens_runtime=torch.tensor(kv_lens),
        tokens_per_block=tokens_per_block,
        num_generations=rows,
    )


def test_sampling_plan_groups_rows_with_common_leading_pages():
    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    tpb = 4
    # Rows 0, 2, 3 share pages 5, 6 (a third page differs); row 1 is alone;
    # rows 2, 3 also share page 7 but row 2's current token sits in it.
    tables = [[5, 6, 10], [20, 21, 22], [5, 6, 7], [5, 6, 7, 11]]
    kv_lens = [10, 9, 12, 14]
    meta = _fake_sampling_metadata(tables, kv_lens, tpb, num_pages=32)
    lib._plan_sampling_step(meta)
    plan = meta.fmha_plan_caches[lib.SAMPLING_KEY].plan
    assert plan is not None

    # Groups in order of first appearance: {0, 2, 3} then {1}.
    assert plan.perm.tolist() == [0, 2, 3, 1]
    assert plan.qo_indptr.tolist() == [0, 3, 4]
    # Row 2's current token (position 11) is in its third block, so the group
    # shares two pages; row 1 has no partner and uses a one-token placeholder.
    assert plan.prefix_indptr.tolist() == [0, 2, 3]
    assert plan.prefix_indices.tolist()[:2] == [5, 6]
    assert plan.prefix_last_page_len.tolist() == [tpb, 1]
    assert plan.prefix_empty.tolist() == [False, False, False, True]
    # Private pages start after the shared ones; the last page length covers
    # the current token.
    assert plan.tail_indptr.tolist() == [0, 1, 4, 5, 7]
    assert plan.tail_indices.tolist() == [10, 20, 21, 22, 7, 7, 11]
    assert plan.tail_last_page_len.tolist() == [2, 1, 4, 2]
    assert plan.write_page.tolist() == [10, 22, 7, 11]
    assert plan.write_slot.tolist() == [1, 0, 3, 1]


def test_sampling_plan_declines_when_no_rows_share_pages():
    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    meta = _fake_sampling_metadata([[1, 2], [3, 4], [5, 6]], [7, 7, 7], 4, 16)
    lib._plan_sampling_step(meta)
    assert meta.fmha_plan_caches[lib.SAMPLING_KEY].plan is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("batch_order", ["grouped", "interleaved"])
def test_sampling_prefix_and_tail_match_dense_attention(batch_order):
    """Prefix call over shared pages plus paged-decode tail equals dense attention."""

    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    device = torch.device("cuda")
    tpb, num_kv_heads, group, head_dim = 16, 2, 4, 128
    num_heads = num_kv_heads * group
    sm_scale = 1.0 / math.sqrt(head_dim)
    gen = torch.Generator(device=device).manual_seed(3)

    # Two prompts, three children each. Pages 0-3 hold prompt A's four full
    # blocks, 4-7 prompt B's; every child owns two private pages after them.
    shared_pages = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}
    tables, kv_lens, prompt_of = [], [], []
    next_page = 8
    for prompt in (0, 1):
        for child in range(3):
            private = [next_page, next_page + 1]
            next_page += 2
            tables.append(shared_pages[prompt] + private)
            kv_lens.append(4 * tpb + 5 + 3 * child + 17 * prompt)
            prompt_of.append(prompt)
    if batch_order == "interleaved":
        order = [0, 3, 1, 4, 2, 5]
        tables = [tables[i] for i in order]
        kv_lens = [kv_lens[i] for i in order]
        prompt_of = [prompt_of[i] for i in order]
    rows = len(tables)

    pool = torch.randn(
        next_page,
        2,
        num_kv_heads,
        tpb,
        head_dim,
        generator=gen,
        device=device,
        dtype=torch.bfloat16,
    )
    meta = _fake_sampling_metadata(tables, kv_lens, tpb, next_page)
    lib._plan_sampling_step(meta)
    state = meta.fmha_plan_caches[lib.SAMPLING_KEY]
    plan = state.plan
    assert plan is not None and plan.num_groups == 2

    q = torch.randn(rows, num_heads, head_dim, generator=gen, device=device, dtype=torch.bfloat16)
    key = lib._PlanKey(num_heads, num_kv_heads, head_dim, torch.bfloat16, sm_scale)
    indices = lib._sampling_device(state, device)
    prefix_w, tail_w = lib._sampling_wrappers(state, key, meta, device)
    q_grouped = q if indices.perm is None else q.index_select(0, indices.perm)
    p_out, p_lse = prefix_w.run(q_grouped.contiguous(), pool, return_lse=True)
    empty = indices.prefix_empty
    if indices.inv_perm is not None:
        p_out = p_out.index_select(0, indices.inv_perm)
        p_lse = p_lse.index_select(0, indices.inv_perm)
        empty = empty.index_select(0, indices.inv_perm)
    t_out, t_lse = tail_w.run(q.contiguous(), pool, return_lse=True)
    merged = lib.merge_states(p_out, p_lse, t_out, t_lse, empty)

    for r in range(rows):
        pages = torch.tensor(tables[r], device=device)
        kv = pool[pages].permute(0, 3, 1, 2, 4).reshape(-1, 2, num_kv_heads, head_dim)[: kv_lens[r]]
        qr = q[r].float().view(num_kv_heads, group, head_dim)
        scores = torch.einsum("kgd,tkd->kgt", qr, kv[:, 0].float()) * sm_scale
        expected = torch.einsum(
            "kgt,tkd->kgd", torch.softmax(scores, dim=-1), kv[:, 1].float()
        ).reshape(num_heads, head_dim)
        torch.testing.assert_close(merged[r], expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_sampling_merge_is_as_accurate_as_one_decode_kernel():
    """Splitting a row's attention into prefix and tail states must not cost
    accuracy over a single paged-decode call on the whole page table.

    The comparison is against float64 attention. The library writes its merged
    output in bf16 like any single kernel, so its error has to stay at the
    level of one bf16 rounding; rounding the two partial states to bf16 before
    the merge adds a second rounding and enough error to flip greedy
    near-ties against the regular decode kernel.
    """
    flashinfer = pytest.importorskip("flashinfer")
    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    device = torch.device("cuda")
    tpb, num_kv_heads, group, head_dim = 32, 8, 2, 128
    num_heads = num_kv_heads * group
    sm_scale = 1.0 / math.sqrt(head_dim)
    gen = torch.Generator(device=device).manual_seed(5)

    # Two prompts of 48 full shared pages, four children each with a private
    # partial prompt page and a page of generated tokens.
    num_shared, children = 48, 4
    tables, kv_lens = [], []
    next_page = 2 * num_shared
    for prompt in (0, 1):
        shared = list(range(prompt * num_shared, (prompt + 1) * num_shared))
        for child in range(children):
            tables.append(shared + [next_page, next_page + 1])
            next_page += 2
            kv_lens.append(num_shared * tpb + tpb + 3 + child + 7 * prompt)
    rows = len(tables)

    pool = torch.randn(
        next_page,
        2,
        num_kv_heads,
        tpb,
        head_dim,
        generator=gen,
        device=device,
        dtype=torch.bfloat16,
    )
    q = torch.randn(rows, num_heads, head_dim, generator=gen, device=device, dtype=torch.bfloat16)
    meta = _fake_sampling_metadata(tables, kv_lens, tpb, next_page)
    lib._plan_sampling_step(meta)
    state = meta.fmha_plan_caches[lib.SAMPLING_KEY]
    assert state.plan is not None and state.plan.num_groups == 2
    key = lib._PlanKey(num_heads, num_kv_heads, head_dim, torch.bfloat16, sm_scale)
    indices = lib._sampling_device(state, device)
    assert indices.perm is None
    prefix_w, tail_w = lib._sampling_wrappers(state, key, meta, device)
    p_out, p_lse = prefix_w.run(q, pool, return_lse=True)
    t_out, t_lse = tail_w.run(q, pool, return_lse=True)
    merged = lib.merge_states(p_out, p_lse, t_out, t_lse, indices.prefix_empty)
    merged = merged.to(torch.bfloat16)

    # One paged-decode call per row over its whole page table.
    num_blocks = [(length + tpb - 1) // tpb for length in kv_lens]
    indptr = torch.zeros(rows + 1, dtype=torch.int32)
    indptr[1:] = torch.cumsum(torch.tensor(num_blocks), dim=0)
    page_ids = torch.tensor(
        [p for table, nb in zip(tables, num_blocks) for p in table[:nb]], dtype=torch.int32
    )
    last = torch.tensor(
        [length - (nb - 1) * tpb for length, nb in zip(kv_lens, num_blocks)], dtype=torch.int32
    )
    single_w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        "HND",
        use_tensor_cores=True,
    )
    single_w.plan(
        indptr,
        page_ids,
        last,
        num_heads,
        num_kv_heads,
        head_dim,
        tpb,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=sm_scale,
    )
    single = single_w.run(q, pool)

    merged_err = single_err = 0.0
    ref_norm = 0.0
    for r in range(rows):
        pages = torch.tensor(tables[r], device=device)
        kv = (
            pool[pages]
            .permute(0, 3, 1, 2, 4)
            .reshape(-1, 2, num_kv_heads, head_dim)[: kv_lens[r]]
            .double()
        )
        qr = q[r].double().view(num_kv_heads, group, head_dim)
        scores = torch.einsum("kgd,tkd->kgt", qr, kv[:, 0]) * sm_scale
        expected = torch.einsum("kgt,tkd->kgd", torch.softmax(scores, dim=-1), kv[:, 1]).reshape(
            num_heads, head_dim
        )
        merged_err += float((merged[r].double() - expected).square().sum())
        single_err += float((single[r].double() - expected).square().sum())
        ref_norm += float(expected.square().sum())
    merged_err = math.sqrt(merged_err / ref_norm)
    single_err = math.sqrt(single_err / ref_norm)
    assert merged_err <= 1.15 * single_err, (merged_err, single_err)


def _run_n_sampling(model_path, prompts, n, max_tokens, env):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        llm = LLM(
            model=str(model_path),
            max_batch_size=len(prompts) * n,
            max_num_tokens=8192,
            max_seq_len=max(len(p) for p in prompts) + max_tokens + 1,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True, use_kv_cache_manager_v2=False, free_gpu_memory_fraction=0.3
            ),
            cuda_graph_config=None,
            skip_tokenizer_init=True,
        )
        with llm:
            sampling = SamplingParams(
                n=n, top_k=1, max_tokens=max_tokens, ignore_eos=True, end_id=-1
            )
            outputs = llm.generate(prompts, sampling_params=sampling)
        return [[tuple(c.token_ids) for c in out.outputs] for out in outputs]
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(_qwen3_small_path() is None, reason="Qwen3-0.6B checkpoint not available")
@pytest.mark.threadleak(enabled=False)
def test_n_sampling_shared_prefix_matches_fallback_e2e():
    """Greedy n-sample requests give the same tokens with and without the library."""
    model_path = _qwen3_small_path()
    vocab = 150000
    prompts = [
        [1024 + ((17 * i + 13 * p) % (vocab - 1024)) for p in range(n)]
        for i, n in enumerate((600, 410))
    ]
    n, max_tokens = 6, 24

    base_env = {
        "TLLM_WORKER_USE_SINGLE_PROCESS": "1",
        "TLLM_ALLOW_N_GREEDY_DECODING": "1",
    }
    reference = _run_n_sampling(model_path, prompts, n, max_tokens, base_env)
    candidate = _run_n_sampling(
        model_path, prompts, n, max_tokens, base_env | {"TLLM_FMHA_LIBS": f"+{LIB_NAME}"}
    )
    for ref_samples, cand_samples in zip(reference, candidate):
        assert len(cand_samples) == n
        # Greedy children are identical; bf16 kernel differences may flip a
        # late token, so require the first tokens to agree.
        for ref_tokens, cand_tokens in zip(ref_samples, cand_samples):
            assert cand_tokens[:8] == ref_tokens[:8]


def test_sampling_plan_reads_kv_manager_v2_block_table():
    """V2 keeps base page indices by request slot; the planner rescales them to
    the pages of the layer's ``get_buffers`` view."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    tpb, index_scale, kv_factor = 4, 4, 2
    # Base page indices per request slot; slot order differs from batch order.
    slot_tables = {
        3: [5, 6, 10],
        0: [5, 6, 7],
        1: [20, 21, 22],
    }
    request_ids = [101, 102, 103]  # batch order -> slots 3, 0, 1
    slot_of = {101: 3, 102: 0, 103: 1}
    max_blocks = 3
    host = torch.zeros(1, 4, 2, max_blocks, dtype=torch.int32)
    for slot, table in slot_tables.items():
        host[0, slot, 0, :] = torch.tensor(table)
    manager = SimpleNamespace(
        layer_offsets={7: 0},
        kv_factor=kv_factor,
        max_beam_width=1,
        index_mapper=SimpleNamespace(get_index=lambda rid: slot_of[rid]),
        layer_to_pool_mapping_dict={0: 0},
        get_layer_page_index_scale=lambda layer_idx: index_scale,
        get_buffers=lambda layer, kv_layout: torch.empty(200, 1),
    )
    meta = SimpleNamespace(
        fmha_plan_caches={},
        kv_cache_manager=manager,
        host_kv_cache_block_offsets=host,
        kv_lens_runtime=torch.tensor([10, 12, 9]),
        tokens_per_block=tpb,
        num_generations=3,
        request_ids=request_ids,
    )
    lib._plan_sampling_step(meta)
    plan = meta.fmha_plan_caches[lib.SAMPLING_KEY].plan
    assert plan is not None
    pages_per_base = index_scale // kv_factor
    # Rows 0 and 1 share two base pages; row 2 stands alone.
    assert plan.perm is None
    assert plan.qo_indptr.tolist() == [0, 2, 3]
    assert plan.prefix_indices.tolist()[:2] == [5 * pages_per_base, 6 * pages_per_base]
    assert plan.tail_indices.tolist() == [
        10 * pages_per_base,
        7 * pages_per_base,
        20 * pages_per_base,
        21 * pages_per_base,
        22 * pages_per_base,
    ]
