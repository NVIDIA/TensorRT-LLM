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


@pytest.fixture(autouse=True)
def _open_size_gate(monkeypatch):
    """The tests use short prompts; open the size gate unless a test sets it."""
    monkeypatch.setenv("TLLM_BEAM_SHARED_PREFIX_MIN_SHARED_TOKENS", "0")
    monkeypatch.setenv("TLLM_BEAM_SHARED_PREFIX_MIN_GROUP", "1")


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


def _fake_sampling_metadata(
    block_tables,
    kv_lens,
    tokens_per_block,
    num_pages,
    *,
    max_blocks=None,
    is_cuda_graph=False,
    plan_caches=None,
):
    """Minimal metadata for the host planner: one pool, page stride 2."""
    from types import SimpleNamespace

    rows = len(block_tables)
    if max_blocks is None:
        max_blocks = max(len(t) for t in block_tables)
    offsets = torch.zeros(1, rows, 2, max_blocks, dtype=torch.int32)
    for r, table in enumerate(block_tables):
        offsets[0, r, 0, : len(table)] = torch.tensor(table) * 2
    manager = SimpleNamespace(
        layer_offsets={0: 0},
        kv_factor=2,
        get_buffers=lambda layer, kv_layout: torch.empty(num_pages, 1),
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return SimpleNamespace(
        fmha_plan_caches={} if plan_caches is None else plan_caches,
        kv_cache_manager=manager,
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]]),
        host_kv_cache_block_offsets=offsets,
        kv_cache_block_offsets=torch.empty((0,), device=device),
        kv_lens_runtime=torch.tensor(kv_lens),
        tokens_per_block=tokens_per_block,
        num_generations=rows,
        is_cuda_graph=is_cuda_graph,
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
    assert plan.has_empty
    # Private pages start after the shared ones, in the grouped row order
    # (rows 0, 2, 3, 1); the last page length covers the current token.
    assert plan.tail_indptr.tolist() == [0, 1, 2, 4, 7]
    assert plan.tail_indices.tolist() == [10, 7, 7, 11, 20, 21, 22]
    assert plan.tail_last_page_len.tolist() == [2, 4, 2, 1]
    assert plan.write_page.tolist() == [10, 22, 7, 11]
    assert plan.write_slot.tolist() == [1, 0, 3, 1]


def test_sharing_pays_gate(monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    monkeypatch.delenv(lib.MIN_SHARED_ENV)
    monkeypatch.delenv(lib.MIN_GROUP_ENV)
    # 4 groups of 20 rows.
    assert lib.sharing_pays(80, 80, 4, 6000)
    assert not lib.sharing_pays(80, 80, 4, 4000)  # prompt too short
    assert not lib.sharing_pays(80, 80, 4, 600)
    assert not lib.sharing_pays(64, 64, 32, 8000)  # groups of 2: too narrow
    assert not lib.sharing_pays(80, 20, 1, 8000)  # most rows do not share
    assert not lib.sharing_pays(8, 0, 0, 0)
    monkeypatch.setenv(lib.MIN_SHARED_ENV, "1000")
    assert lib.sharing_pays(80, 80, 4, 1600)
    monkeypatch.setenv(lib.MIN_GROUP_ENV, "30")
    assert not lib.sharing_pays(80, 80, 4, 1600)


def test_sampling_plan_applies_size_gate(monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    tpb = 4
    tables = [[5, 6, 10], [5, 6, 11], [5, 6, 12], [5, 6, 13]]
    kv_lens = [10, 10, 10, 10]
    # Eight shared tokens per row in one group of four.
    meta = _fake_sampling_metadata(tables, kv_lens, tpb, num_pages=32)
    lib._plan_sampling_step(meta)
    assert meta.fmha_plan_caches[lib.SAMPLING_KEY].plan is not None
    monkeypatch.setenv(lib.MIN_SHARED_ENV, "9")
    meta = _fake_sampling_metadata(tables, kv_lens, tpb, num_pages=32)
    lib._plan_sampling_step(meta)
    assert meta.fmha_plan_caches[lib.SAMPLING_KEY].plan is None
    monkeypatch.setenv(lib.MIN_SHARED_ENV, "8")
    monkeypatch.setenv(lib.MIN_GROUP_ENV, "5")
    meta = _fake_sampling_metadata(tables, kv_lens, tpb, num_pages=32)
    lib._plan_sampling_step(meta)
    assert meta.fmha_plan_caches[lib.SAMPLING_KEY].plan is None


def test_resolve_graph_variant_follows_gate(monkeypatch):
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    def request(rid, parent, prompt_len):
        return SimpleNamespace(
            py_request_id=rid,
            parent_request_id=parent,
            is_child=parent is not None,
            py_prompt_len=prompt_len,
        )

    def batch(prompt_len, n, patients):
        reqs = []
        for p in range(patients):
            reqs.append(request(100 * p, None, prompt_len))
            reqs += [request(100 * p + i, 100 * p, prompt_len) for i in range(1, n)]
        return SimpleNamespace(generation_requests=reqs)

    monkeypatch.delenv(lib.MIN_SHARED_ENV)
    monkeypatch.delenv(lib.MIN_GROUP_ENV)
    assert lib.resolve_graph_variant(batch(6000, 20, 4)) == 1
    assert lib.resolve_graph_variant(batch(1600, 20, 4)) == 0
    assert lib.resolve_graph_variant(batch(8000, 1, 16)) == 0  # no sharing
    # Padding dummies are singleton rows: still library when most rows share.
    mixed = batch(6000, 20, 4)
    mixed.generation_requests.append(request(9999, None, 128))
    assert lib.resolve_graph_variant(mixed) == 1
    assert lib.resolve_graph_variant(SimpleNamespace(generation_requests=[])) == 0


def test_graph_key_carries_attention_variant():
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType

    assert KeyType(batch_size=8, draft_len=0) != KeyType(batch_size=8, draft_len=0, attn_variant=1)
    assert KeyType(batch_size=8, draft_len=0).attn_variant == 0


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
    p_out, p_lse = prefix_w.run(q_grouped, pool, return_lse=True)
    t_out, t_lse = tail_w.run(q_grouped, pool, return_lse=True)
    merged_grouped = lib.merge_states(p_out, p_lse, t_out, t_lse, indices.prefix_empty)
    # Back to batch order for the comparison with the per-row reference.
    merged = merged_grouped
    if indices.perm is not None:
        merged = torch.empty_like(merged_grouped)
        merged[indices.perm] = merged_grouped

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


def _run_n_sampling(model_path, prompts, n, max_tokens, env, cuda_graph=False):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

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
            cuda_graph_config=(
                CudaGraphConfig(max_batch_size=len(prompts) * n) if cuda_graph else None
            ),
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
@pytest.mark.parametrize("cuda_graph", [False, True], ids=["eager", "cuda_graph"])
@pytest.mark.parametrize("gate_tokens", ["0", "100000"], ids=["library", "gated_off"])
def test_n_sampling_shared_prefix_matches_fallback_e2e(cuda_graph, gate_tokens):
    """Greedy n-sample requests give the same tokens with and without the
    library, eager and with every decode step replayed from a CUDA graph. With
    the gate closed the library declines every batch (CUDA graphs capture the
    regular-kernel variant); with it open the library variant runs."""
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
    reference = _run_n_sampling(model_path, prompts, n, max_tokens, base_env, cuda_graph)
    candidate = _run_n_sampling(
        model_path,
        prompts,
        n,
        max_tokens,
        base_env
        | {
            "TLLM_FMHA_LIBS": f"+{LIB_NAME}",
            "TLLM_BEAM_SHARED_PREFIX_MIN_SHARED_TOKENS": gate_tokens,
            "TLLM_BEAM_SHARED_PREFIX_MIN_GROUP": "1",
        },
        cuda_graph,
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
        is_cuda_graph=False,
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


def test_sampling_grouping_pads_groups_and_demotes_surplus_sharing_groups():
    """A CUDA graph batch gets a fixed number of prefix groups: the sharing
    groups that save the most page reads, the placeholder group (also without
    rows) and empty padding groups."""
    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    tpb = 4
    # Group {0, 1} shares two pages, group {2, 3, 4} shares one page (the
    # current token of row 4 sits in the second), row 5 shares nothing.
    tables = [[1, 2, 9], [1, 2, 10], [3, 4, 11], [3, 4, 12], [3, 13], [5, 14]]
    kv_lens = torch.tensor([10, 11, 9, 12, 7, 6])
    pages = torch.tensor([t + [0] * (3 - len(t)) for t in tables], dtype=torch.int64)
    rows = len(tables)

    grouping = lib._build_grouping(
        1,
        torch.arange(rows),
        pages,
        kv_lens,
        tpb,
        max_sharing_groups=None,
        padded_groups=4,
        keep_check_pages=False,
    )
    assert grouping.num_sharing == 2 and grouping.num_groups == 4
    assert grouping.perm.tolist() == [0, 1, 2, 3, 4, 5]
    assert grouping.qo_indptr.tolist() == [0, 2, 5, 6, 6]
    assert grouping.prefix_indptr.tolist() == [0, 2, 3, 4, 5]
    assert grouping.prefix_indices.tolist()[:3] == [1, 2, 3]
    assert grouping.prefix_last_page_len.tolist() == [tpb, tpb, 1, 1]
    assert grouping.prefix_empty.tolist() == [False] * 5 + [True]

    # Room for one sharing group: {0, 1} and {2, 3, 4} both save two page
    # reads; the tie goes to the group with the lower first page.
    grouping = lib._build_grouping(
        2,
        torch.arange(rows),
        pages,
        torch.tensor([10, 11, 9, 12, 7, 6]),
        tpb,
        max_sharing_groups=1,
        padded_groups=2,
        keep_check_pages=False,
    )
    assert grouping.num_sharing == 1
    assert grouping.perm.tolist() == [0, 1, 2, 3, 4, 5]
    assert grouping.qo_indptr.tolist() == [0, 2, 6]
    assert grouping.prefix_empty.tolist() == [False, False] + [True] * 4
    # Demoted rows read all their pages through the tail call.
    assert grouping.tail_indptr.tolist() == [0, 1, 2, 5, 8, 10, 12]

    # No sharing at all still yields the placeholder group plus padding.
    grouping = lib._build_grouping(
        3,
        torch.arange(rows),
        pages[[0, 2, 5]],
        kv_lens[[0, 2, 5]],
        tpb,
        max_sharing_groups=2,
        padded_groups=3,
        keep_check_pages=False,
    )
    assert grouping.num_sharing == 0
    assert grouping.qo_indptr.tolist() == [0, 3, 3, 3]
    assert grouping.prefix_empty.all()


def test_sampling_tail_refresh_matches_full_regrouping():
    """Rows crossing a page boundary only rebuild the tail tables, unless a
    group's shared run was limited by its members' full pages."""
    import numpy as np

    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    tpb = 4
    # {0, 1, 2} share pages 1-3 and diverge at their fourth page; {3, 4} share
    # page 5 only up to row 4's current page; row 5 stands alone.
    tables = [
        [1, 2, 3, 10, 11, 12],
        [1, 2, 3, 20, 21, 22],
        [1, 2, 3, 30, 31, 32],
        [5, 6, 40, 41, 42, 43],
        [5, 6, 50, 51, 52, 53],
        [7, 60, 61, 62, 63, 64],
    ]
    pages = np.array(tables, dtype=np.int64)
    rows = len(tables)

    def build(kv_lens, version):
        return lib._build_grouping(
            version,
            np.arange(rows),
            pages,
            np.array(kv_lens),
            tpb,
            max_sharing_groups=None,
            padded_groups=4,
            keep_check_pages=False,
        )

    before = [15, 16, 14, 13, 6, 9]
    grouping = build(before, 1)
    assert grouping.num_sharing == 2 and grouping.capped_keys.tolist() == [1]

    # Rows 0 and 5 cross a boundary; group {3, 4} keeps its cap.
    after = [17, 16, 14, 13, 6, 10]
    refreshed = lib._refresh_tail(grouping, 2, pages, np.array(after), tpb, keep_check_pages=False)
    rebuilt = build(after, 2)
    assert refreshed is not None
    assert refreshed.prefix_version == 1 and refreshed.version == 2
    for name in ("perm", "prefix_empty", "write_page", "blocks_g", "private_from"):
        assert getattr(refreshed, name).tolist() == getattr(rebuilt, name).tolist(), name
    for name in (
        "qo_indptr",
        "prefix_indptr",
        "prefix_indices",
        "prefix_last_page_len",
        "tail_indptr",
        "tail_indices",
    ):
        assert torch.equal(getattr(refreshed, name), getattr(rebuilt, name)), name

    # Row 4's current token moves to its third page: group {3, 4} may now share
    # its second page, so the groups have to be rebuilt.
    assert (
        lib._refresh_tail(
            grouping, 3, pages, np.array([15, 16, 14, 13, 9, 9]), tpb, keep_check_pages=False
        )
        is None
    )


def _dense_attention(pool, table, kv_len, q_row, num_kv_heads, sm_scale):
    """fp32 attention of one query row over the first ``kv_len`` tokens of its pages."""
    num_heads, head_dim = q_row.shape
    group = num_heads // num_kv_heads
    pages = torch.tensor(table, device=pool.device)
    kv = pool[pages].permute(0, 3, 1, 2, 4).reshape(-1, 2, num_kv_heads, head_dim)[:kv_len]
    qr = q_row.float().view(num_kv_heads, group, head_dim)
    scores = torch.einsum("kgd,tkd->kgt", qr, kv[:, 0].float()) * sm_scale
    return torch.einsum("kgt,tkd->kgd", torch.softmax(scores, dim=-1), kv[:, 1].float()).reshape(
        num_heads, head_dim
    )


def _sampling_step_tables(prompts, children, private_pages, first_private, gen_lens, tpb):
    """Block tables and KV lengths of ``children`` samples per prompt.

    ``prompts`` lists (first page, prompt length) per prompt; the full prompt
    pages are shared, everything after them is private to the child.
    """
    tables, kv_lens = [], []
    next_page = first_private
    for (first, prompt_len), count in zip(prompts, children):
        full = prompt_len // tpb
        for child in range(count):
            kv_len = prompt_len + gen_lens[len(kv_lens) % len(gen_lens)] + child % 3
            num_blocks = (kv_len + tpb - 1) // tpb
            private = list(range(next_page, next_page + num_blocks - full))
            next_page += private_pages
            tables.append(list(range(first, first + full)) + private)
            kv_lens.append(kv_len)
    return tables, kv_lens


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("num_heads, num_kv_heads", [(8, 2), (20, 20), (20, 4)])
def test_sampling_cuda_graph_replay_matches_eager_across_steps(
    num_heads, num_kv_heads, monkeypatch
):
    """Capture one CUDA graph around the n-sample attention of a layer and
    replay it over steps whose plans differ: tails that grow inside a page and
    across a page boundary, shrinking groups, different group counts, surplus
    sharing groups, a batch without any sharing, and padded dummy rows.

    Each replay must equal an eager call on the same planned buffers bit for
    bit (the graph reads the refreshed buffers, not the captured ones), match
    a dense fp32 reference, and write the current token's K/V like the eager
    call.
    """
    pytest.importorskip("flashinfer")
    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

    monkeypatch.setenv(lib.MAX_GROUPS_ENV, "3")
    device = torch.device("cuda")
    tpb, head_dim = 16, 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    rows, num_pages, max_blocks = 12, 160, 12
    gen = torch.Generator(device=device).manual_seed(11)
    pool = torch.randn(
        num_pages,
        2,
        num_kv_heads,
        tpb,
        head_dim,
        generator=gen,
        device=device,
        dtype=torch.bfloat16,
    )

    # Prompt pages 0-29; private pages from 40 on, 8 per row.
    steps = []
    # 1. Three prompts x four children with ragged prompts and tails.
    steps.append(
        _sampling_step_tables([(0, 70), (10, 37), (20, 100)], [4, 4, 4], 8, 40, [1, 3], tpb)
    )
    # 2. Same rows, every tail one token longer (same pages).
    steps.append((steps[0][0], [n + 1 for n in steps[0][1]]))
    # 3. Tails across a page boundary.
    steps.append(
        _sampling_step_tables([(0, 70), (10, 37), (20, 100)], [4, 4, 4], 8, 40, [12, 14], tpb)
    )
    # 4. Fewer, smaller groups: 2 x 3 children, the rest unrelated prompts.
    tables, kv_lens = _sampling_step_tables([(0, 70), (20, 100)], [3, 3], 8, 40, [5], tpb)
    for r in range(6):
        tables.append([30 + r] + list(range(100 + 8 * r, 104 + 8 * r)))
        kv_lens.append(40 + r)
    steps.append((tables, kv_lens))
    # 5. Four sharing groups for three group slots: the smallest saving is demoted.
    steps.append(
        _sampling_step_tables(
            [(0, 70), (10, 37), (20, 100), (5, 17)], [3, 3, 3, 3], 8, 40, [2], tpb
        )
    )
    # 6. No sharing at all.
    steps.append(
        (
            [[30 + r, 100 + 4 * r, 101 + 4 * r] for r in range(rows)],
            [33 + r for r in range(rows)],
        )
    )
    # 7. Two real groups and padded dummy rows that all map one table (kv_len 1).
    tables, kv_lens = _sampling_step_tables([(0, 70), (10, 37)], [4, 4], 8, 40, [2], tpb)
    for _ in range(4):
        tables.append([150])
        kv_lens.append(1)
    steps.append((tables, kv_lens))

    # (sharing groups, rows without a shared prefix) of every step.
    expected_groups = [(3, 0), (3, 0), (3, 0), (2, 6), (3, 3), (0, 12), (2, 4)]
    caches = {}
    key = lib._PlanKey(num_heads, num_kv_heads, head_dim, torch.bfloat16, sm_scale)
    q_static = torch.zeros(rows, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_static = torch.zeros(rows, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_static = torch.zeros_like(k_static)
    out_static = torch.zeros_like(q_static)
    # Device KV lengths at a fixed address, refreshed per step like the
    # metadata's kv_lens_cuda_runtime that the graph reads.
    kv_lens_static = torch.zeros(rows, dtype=torch.int32, device=device)
    graph = None
    previous_version = None
    state = None
    wrappers = None
    for step, (tables, kv_lens) in enumerate(steps):
        assert len(tables) == rows
        meta = _fake_sampling_metadata(
            tables,
            kv_lens,
            tpb,
            num_pages,
            max_blocks=max_blocks,
            is_cuda_graph=True,
            plan_caches=caches,
        )
        lib._plan_sampling_step(meta)
        state = caches[lib.SAMPLING_KEY]
        assert state.plan is not None and state.graph is not None
        grouping = state.plan.grouping
        assert (grouping.num_sharing, grouping.num_placeholder_rows) == expected_groups[step]
        if step == 1:
            # Same pages as step 0: nothing is uploaded or re-planned.
            assert grouping.version == previous_version
        previous_version = grouping.version
        assert grouping.num_groups == state.graph.num_groups == 4
        if wrappers is None:
            wrappers = lib._graph_wrappers(state, key, meta)
        else:
            assert lib._graph_wrappers(state, key, meta) == wrappers
        buffers = state.graph

        q = torch.randn(rows, num_heads, head_dim, generator=gen, device=device).bfloat16()
        k = torch.randn(rows, num_kv_heads, head_dim, generator=gen, device=device).bfloat16()
        v = torch.randn(rows, num_kv_heads, head_dim, generator=gen, device=device).bfloat16()
        dummy_rows = [r for r in range(rows) if kv_lens[r] == 1]
        for r in dummy_rows:
            # Padding rows repeat one request: identical inputs, one KV slot.
            q[r], k[r], v[r] = q[dummy_rows[0]], k[dummy_rows[0]], v[dummy_rows[0]]

        kv_lens_static.copy_(torch.tensor(kv_lens, dtype=torch.int32))

        def run(target_pool, out, q_in, k_in, v_in):
            # Between page boundaries prepare() uploads nothing: the graph
            # derives the write slots and last-page lengths on the device.
            lib.step_slots(
                kv_lens_static,
                buffers.perm,
                buffers.write_slot,
                buffers.tail_last_page_len,
                tpb,
            )
            lib.sampling_attention(
                target_pool,
                q_in,
                k_in,
                v_in,
                out,
                *wrappers,
                write_page=buffers.write_page,
                write_slot=buffers.write_slot,
                perm=buffers.perm,
                prefix_empty=buffers.empty,
            )

        eager_pool = pool.clone()
        eager_out = torch.zeros_like(q)
        run(eager_pool, eager_out, q, k, v)

        if graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run(pool, out_static, q_static, k_static, v_static)
        q_static.copy_(q)
        k_static.copy_(k)
        v_static.copy_(v)
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(out_static, eager_out), f"step {step}: replay differs from eager"
        assert torch.equal(pool, eager_pool), f"step {step}: replay wrote different K/V"
        for r in range(rows):
            if r in dummy_rows:
                continue
            expected = _dense_attention(pool, tables[r], kv_lens[r], q[r], num_kv_heads, sm_scale)
            torch.testing.assert_close(
                out_static[r].float(), expected, atol=2e-2, rtol=2e-2, msg=f"step {step} row {r}"
            )
        # An independently planned eager batch agrees within kernel numerics.
        eager_meta = _fake_sampling_metadata(tables, kv_lens, tpb, num_pages)
        lib._plan_sampling_step(eager_meta)
        eager_state = eager_meta.fmha_plan_caches[lib.SAMPLING_KEY]
        if eager_state.plan is not None:
            eager_indices = lib._sampling_device(eager_state, device)
            eager_wrappers = lib._sampling_wrappers(eager_state, key, eager_meta, device)
            reference_pool = eager_pool.clone()
            reference_out = torch.zeros_like(q)
            lib.sampling_attention(
                reference_pool,
                q,
                k,
                v,
                reference_out,
                *eager_wrappers,
                write_page=eager_indices.write_page,
                write_slot=eager_indices.write_slot,
                perm=eager_indices.perm,
                prefix_empty=eager_indices.prefix_empty,
            )
            live = [r for r in range(rows) if r not in dummy_rows]
            torch.testing.assert_close(
                out_static[live].float(), reference_out[live].float(), atol=1e-2, rtol=1e-2
            )
