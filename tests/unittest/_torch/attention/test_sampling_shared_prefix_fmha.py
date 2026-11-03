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
"""Tests for the opt-in sampling_shared_prefix FMHA library (n-sample path, KV cache manager V2).

The planner and the merge math are covered against dense references; the
end-to-end tests run Qwen3-0.6B n-sampling with and without the library and
require a GPU plus the model checkpoint.
"""

import math
import os

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix import LIB_NAME

_LOG2E = 1.4426950408889634


@pytest.fixture(autouse=True)
def _open_size_gate(monkeypatch):
    """The tests use short prompts; open the size gate unless a test sets it."""
    monkeypatch.setenv("TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS", "0")
    monkeypatch.setenv("TLLM_SAMPLING_SHARED_PREFIX_SPEC_MIN_SHARED_TOKENS", "0")
    monkeypatch.setenv("TLLM_SAMPLING_SHARED_PREFIX_SPEC_MIN_ROWS", "1")
    monkeypatch.setenv("TLLM_SAMPLING_SHARED_PREFIX_MIN_GROUP", "1")


@pytest.fixture(autouse=True)
def _no_model_shape(monkeypatch):
    """The library remembers the attention shape of the last model it served; a test that
    plans a fake batch must not pick up the shape of an end-to-end test that ran before it."""
    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

    monkeypatch.setattr(lib, "_model_shape", None)


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


def _qwen3_small_path():
    root = llm_models_root(check=False)
    if root is None:
        return None
    path = root / "Qwen3" / "Qwen3-0.6B"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# n-sample (beam_width == 1) path
# ---------------------------------------------------------------------------
def test_merge_states_matches_dense_attention():
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix import merge_states

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
    q_len=1,
    device_kv_lens=None,
):
    """Minimal metadata for the host planner on a V2-style manager: one pool whose
    base page indices are scaled by 2 into the K/V-folded pool view (page == entry).

    ``q_len`` tokens per row (speculative decoding); ``device_kv_lens`` are the
    true lengths when they differ from the host ``kv_lens``."""
    from types import SimpleNamespace

    rows = len(block_tables)
    if max_blocks is None:
        max_blocks = max(len(t) for t in block_tables)
    offsets = torch.zeros(1, rows, 2, max_blocks, dtype=torch.int32)
    for r, table in enumerate(block_tables):
        offsets[0, r, 0, : len(table)] = torch.tensor(table)

    class _Manager(SimpleNamespace):  # weak-referenceable, like the real managers
        pass

    manager = _Manager(
        layer_offsets={0: 0},
        kv_factor=2,
        max_beam_width=1,
        index_mapper=SimpleNamespace(get_index=lambda rid: rid),
        layer_to_pool_mapping_dict={0: 0},
        get_layer_page_index_scale=lambda layer_idx: 2,
        get_buffers=lambda layer, kv_layout: torch.empty(num_pages, 1),
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return SimpleNamespace(
        fmha_plan_caches={} if plan_caches is None else plan_caches,
        kv_cache_manager=manager,
        request_ids=list(range(rows)),
        host_kv_cache_block_offsets=offsets,
        kv_cache_block_offsets=torch.empty((0,), device=device),
        kv_lens_runtime=torch.tensor(kv_lens),
        kv_lens_cuda_runtime=torch.tensor(
            kv_lens if device_kv_lens is None else device_kv_lens, dtype=torch.int32, device=device
        ),
        tokens_per_block=tokens_per_block,
        num_generations=rows,
        num_tokens=rows * q_len,
        seq_lens=torch.full((rows,), q_len, dtype=torch.int32),
        is_cuda_graph=is_cuda_graph,
    )


def test_sampling_plan_groups_rows_with_common_leading_pages():
    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

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
    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

    monkeypatch.delenv(lib.MIN_SHARED_ENV)
    monkeypatch.delenv(lib.MIN_GROUP_ENV)
    monkeypatch.setattr(lib, "_model_shape", None)  # layer shape unknown: the fixed default
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


def test_resolve_graph_variant_follows_gate(monkeypatch):
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

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
    # Captured speculative steps (draft tokens) that the fused kernel cannot serve use
    # the regular kernel even when the speculative size gate is open (FlashInfer's
    # captured split-KV tail plan follows the host lengths and is wrong when the device
    # lengths are shorter). With a supported layer shape they follow the gate.
    monkeypatch.setenv(lib.SPEC_MIN_SHARED_ENV, "0")
    spec = batch(6000, 20, 4)
    for r in spec.generation_requests:
        r.py_draft_tokens = [1, 2, 3]
    monkeypatch.setattr(lib, "_model_shape", None)
    assert lib.resolve_graph_variant(spec) == 0
    monkeypatch.setattr(lib, "_model_shape", (20, 20, 128))
    assert lib.resolve_graph_variant(spec) == 1


def test_sampling_plan_declines_when_no_rows_share_pages():
    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

    meta = _fake_sampling_metadata([[1, 2], [3, 4], [5, 6]], [7, 7, 7], 4, 16)
    lib._plan_sampling_step(meta)
    assert meta.fmha_plan_caches[lib.SAMPLING_KEY].plan is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("batch_order", ["interleaved"])
def test_sampling_prefix_and_tail_match_dense_attention(batch_order):
    """Prefix call over shared pages plus paged-decode tail equals dense attention."""

    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

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
    indices = lib._sampling_device(state, device, meta)
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


def _run_n_sampling(model_path, prompts, n, max_tokens, env, cuda_graph=False, with_logprobs=False):
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
                enable_block_reuse=True, use_kv_cache_manager_v2=True, free_gpu_memory_fraction=0.3
            ),
            cuda_graph_config=(
                CudaGraphConfig(max_batch_size=len(prompts) * n) if cuda_graph else None
            ),
            skip_tokenizer_init=True,
        )
        with llm:
            sampling = SamplingParams(
                n=n,
                top_k=1,
                max_tokens=max_tokens,
                ignore_eos=True,
                end_id=-1,
                logprobs=2 if with_logprobs else None,
            )
            outputs = llm.generate(prompts, sampling_params=sampling)
        if with_logprobs:
            return [
                [
                    (
                        tuple(c.token_ids),
                        [sorted((v.logprob for v in d.values()), reverse=True) for d in c.logprobs],
                    )
                    for c in out.outputs
                ]
                for out in outputs
            ]
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
@pytest.mark.parametrize("cuda_graph", [True], ids=["cuda_graph"])
@pytest.mark.parametrize("gate_tokens", ["100000"], ids=["gated_off"])
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
            "TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS": gate_tokens,
            "TLLM_SAMPLING_SHARED_PREFIX_MIN_GROUP": "1",
        },
        cuda_graph,
    )
    for ref_samples, cand_samples in zip(reference, candidate):
        assert len(cand_samples) == n
        # Greedy children are identical; bf16 kernel differences may flip a
        # late token, so require the first tokens to agree.
        for ref_tokens, cand_tokens in zip(ref_samples, cand_samples):
            assert cand_tokens[:8] == ref_tokens[:8]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(_qwen3_small_path() is None, reason="Qwen3-0.6B checkpoint not available")
@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("cuda_graph", [True], ids=["cuda_graph"])
def test_n_sampling_shared_prefix_diverges_only_at_near_ties_e2e(cuda_graph):
    """The library is a different bf16 rounding path, not bit-identical to the
    regular kernel, so a greedy child may leave the regular kernel's sequence.
    That may only happen where the regular kernel's own top-2 logprob gap is
    within two bf16 steps of the logits (0.25 nats at logits of 16-32); any
    divergence at a clear margin is a real error (wrong page, position or
    merge), not rounding. Prompts end at and around page boundaries."""
    model_path = _qwen3_small_path()
    vocab = 150000
    # 32-token pages: 31, 32 and 33 tokens past a page multiple.
    prompts = [
        [1024 + ((17 * i + 13 * p) % (vocab - 1024)) for p in range(n)]
        for i, n in enumerate((511, 512, 513))
    ]
    n, max_tokens = 6, 40
    base_env = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1", "TLLM_ALLOW_N_GREEDY_DECODING": "1"}
    reference = _run_n_sampling(
        model_path, prompts, n, max_tokens, base_env, cuda_graph, with_logprobs=True
    )
    candidate = _run_n_sampling(
        model_path,
        prompts,
        n,
        max_tokens,
        base_env
        | {
            "TLLM_FMHA_LIBS": f"+{LIB_NAME}",
            "TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS": "0",
            "TLLM_SAMPLING_SHARED_PREFIX_MIN_GROUP": "1",
        },
        cuda_graph,
        with_logprobs=True,
    )
    for ref_samples, cand_samples in zip(reference, candidate):
        assert len(cand_samples) == n
        for (ref_tokens, ref_lp), (cand_tokens, _) in zip(ref_samples, cand_samples):
            assert len(cand_tokens) == len(ref_tokens)
            if cand_tokens == ref_tokens:
                continue
            pos = next(i for i, (a, b) in enumerate(zip(ref_tokens, cand_tokens)) if a != b)
            gap = ref_lp[pos][0] - ref_lp[pos][1] if len(ref_lp[pos]) > 1 else 0.0
            assert gap <= 0.25, (pos, gap)


def test_sampling_plan_reads_kv_manager_v2_block_table():
    """V2 keeps base page indices by request slot; the planner rescales them to
    the pages of the layer's ``get_buffers`` view."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

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

    class _Manager(SimpleNamespace):  # weak-referenceable, like the real managers
        pass

    manager = _Manager(
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
        num_tokens=3,
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
    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

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

    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

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
@pytest.mark.parametrize("num_heads, num_kv_heads", [(20, 20)])
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
    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

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
            eager_indices = lib._sampling_device(eager_state, device, eager_meta)
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


# --------------------------------------------------------------------------- #
# Speculative decoding: q_len = 1 + draft tokens per row
# --------------------------------------------------------------------------- #


def test_beam_search_batches_are_declined():
    """The library serves n-sample batches only: with ``beam_width > 1`` it plans
    nothing and the regular kernel runs."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

    pytest.importorskip("flashinfer")
    manager = SimpleNamespace(
        get_buffers=lambda layer, kv_layout: None,
        host_kv_cache_block_offsets=None,
        kv_factor=2,
        tokens_per_block=32,
        index_mapper=object(),
    )
    meta = SimpleNamespace(
        beam_width=4,
        num_generations=8,
        num_contexts=0,
        is_cross=False,
        enable_helix=False,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        kv_cache_block_offsets=torch.empty(0),
        kv_cache_params=SimpleNamespace(use_cache=True, num_extra_kv_tokens=0),
        kv_cache_manager=manager,
        fmha_plan_caches={},
        is_cuda_graph=False,
    )
    assert not lib.sampling_metadata_eligible(meta)
    lib.plan_step(meta)
    assert meta.fmha_plan_caches == {}
    assert not lib.has_active_plan(meta)
    meta.beam_width = 1
    assert lib.sampling_metadata_eligible(meta)


def test_sampling_declines_a_v1_kv_cache_manager():
    """The n-sample path needs the V2 manager; a V1 manager (no index mapper) is
    declined so that the regular kernel serves the batch."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

    pytest.importorskip("flashinfer")
    v1_manager = SimpleNamespace(
        get_buffers=lambda layer, kv_layout: None,
        host_kv_cache_block_offsets=None,
        kv_factor=2,
        tokens_per_block=32,
    )
    meta = SimpleNamespace(
        beam_width=1,
        num_generations=8,
        num_contexts=0,
        is_cross=False,
        enable_helix=False,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        kv_cache_block_offsets=torch.empty(0),
        kv_cache_params=SimpleNamespace(use_cache=True, num_extra_kv_tokens=0),
        kv_cache_manager=v1_manager,
        fmha_plan_caches={},
        is_cuda_graph=False,
    )
    assert not lib.sampling_metadata_eligible(meta)
    lib.plan_step(meta)  # declines: no n-sample plan, no error
    assert lib.SAMPLING_KEY not in meta.fmha_plan_caches
    assert not lib.has_active_plan(meta)
    v1_manager.index_mapper = object()
    assert lib.sampling_metadata_eligible(meta)


def test_spec_gate_judges_groups_not_the_mean(monkeypatch):
    """With draft tokens the gate sums a convex per-group saving: batches whose groups have very
    different prompt lengths decline even when their mean length is above the homogeneous break-even
    (4x32 and 6x20 rows of mixed lengths measured -5.3% / -6.7% with the mean-length gate)."""
    from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib

    monkeypatch.delenv(lib.SPEC_MIN_SHARED_ENV, raising=False)
    monkeypatch.delenv(lib.MIN_SHARED_ENV, raising=False)
    monkeypatch.delenv(lib.MIN_GROUP_ENV, raising=False)
    monkeypatch.delenv(lib.SPEC_MIN_ROWS_ENV, raising=False)
    monkeypatch.setattr(lib, "_model_shape", (20, 20, 128))
    monkeypatch.setattr(lib, "_ctx_prefix_enabled", lambda head_dim, q_len: True)

    def pays(profile):
        rows = sum(r for r, _ in profile)
        mean = sum(r * t for r, t in profile) / rows
        return lib.sharing_pays(rows, rows, len(profile), mean, q_len=9, group_profile=profile)

    assert not pays([(40, 600)] * 3)  # the 3x40 case at 600 tokens: -26 us per layer
    assert pays([(40, 1600)] * 3)  # +25 us
    assert pays([(40, 4000)] * 3) and pays([(20, 4000)] * 6)
    assert pays([(32, 1600)] * 4)
    assert not pays([(40, 154), (40, 2324), (40, 600)])  # mean 1026
    assert not pays([(32, 937), (32, 119), (32, 384), (32, 3003)])  # mean 1111
    assert not pays(
        [(20, 1542), (20, 446), (20, 233), (20, 86), (20, 4159), (20, 805)]
    )  # mean 1212
    # The three Nsight points the model was solved from.
    saving = lib.spec_batch_saving_us
    assert -35 < saving([(40, 600)] * 3, 120) < -15
    assert 15 < saving([(40, 1600)] * 3, 120) < 35
    assert 150 < saving([(40, 4000)] * 3, 120) < 190
    # An explicit env threshold still wins over the model.
    monkeypatch.setenv(lib.SPEC_MIN_SHARED_ENV, "100")
    assert pays([(40, 600)] * 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,shuffled,placeholder_rows",
    [(16, 8, 128, True, 3), (8, 8, 64, False, 0)],
)
def test_fused_sampling_attention_matches_dense(
    num_heads, num_kv_heads, head_dim, shuffled, placeholder_rows
):
    """The fused prefix + tail + merge kernel against a dense fp32 reference."""
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fused import (
        FusedWorkspace,
        fused_sampling_attention,
    )

    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(1)
    tpb, groups, per_group, prefix_pages = 32, 3, 5, 4
    prefix_len = prefix_pages * tpb
    sharing_rows = groups * per_group
    rows = sharing_rows + placeholder_rows
    gs = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    cached = torch.randint(0, 80, (rows,), generator=gen).tolist()
    # Placeholder rows own every page: a private prompt of 70 tokens plus generated ones.
    cached = [c if r < sharing_rows else c + 70 for r, c in enumerate(cached)]
    own_blocks = [(c + 1 + tpb - 1) // tpb for c in cached]
    num_pages = groups * prefix_pages + 1 + sum(own_blocks)
    pool = torch.randn(num_pages, 2, num_kv_heads, tpb, head_dim, device=device).bfloat16()
    q = torch.randn(rows, num_heads, head_dim, device=device).bfloat16()
    k = torch.randn(rows, num_kv_heads, head_dim, device=device).bfloat16()
    v = torch.randn(rows, num_kv_heads, head_dim, device=device).bfloat16()
    perm = torch.randperm(rows, generator=gen) if shuffled else torch.arange(rows)
    placeholder_page = groups * prefix_pages
    page = placeholder_page + 1
    own = []
    for blocks in own_blocks:
        own.append(list(range(page, page + blocks)))
        page += blocks
    last = [c + 1 - (len(o) - 1) * tpb for c, o in zip(cached, own)]

    def i32(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    qo = [g * per_group for g in range(groups + 1)]
    qo.append(rows if placeholder_rows else qo[-1])
    prefix_indptr = [g * prefix_pages for g in range(groups + 1)] + [groups * prefix_pages + 1]
    prefix_indices = list(range(groups * prefix_pages)) + [placeholder_page]
    prefix_last = [tpb] * groups + [1]
    tail_indptr = [0]
    for o in own:
        tail_indptr.append(tail_indptr[-1] + len(o))
    write_page = torch.empty(rows, dtype=torch.int64)
    write_slot = torch.empty(rows, dtype=torch.int64)
    write_page[perm] = torch.tensor([o[-1] for o in own])
    write_slot[perm] = torch.tensor([x - 1 for x in last])
    empty = torch.tensor([False] * sharing_rows + [True] * placeholder_rows, device=device)
    out = torch.empty(rows, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    pool_before = pool.clone()
    fused_sampling_attention(
        pool,
        q,
        k,
        v,
        out,
        perm=perm.to(device, torch.int64),
        group_qo_indptr=i32(qo),
        prefix_indptr=i32(prefix_indptr),
        prefix_indices=i32(prefix_indices),
        prefix_last_page_len=i32(prefix_last),
        prefix_empty=empty,
        tail_indptr=i32(tail_indptr),
        tail_indices=i32([p for o in own for p in o]),
        tail_last_page_len=i32(last),
        write_page=write_page.to(device),
        write_slot=write_slot.to(device),
        sm_scale=scale,
        num_groups=groups + 1,
        max_group_rows=per_group,
        workspace=FusedWorkspace(rows, num_heads, head_dim, 5, device),
        prefix_splits=4,
        tail_splits=1,
    )
    torch.cuda.synchronize()
    poolf = pool_before.float()
    for i in range(rows):
        b = int(perm[i])
        segments = []
        if i < sharing_rows:
            g = i // per_group
            segments.append(poolf[g * prefix_pages : (g + 1) * prefix_pages].permute(1, 2, 0, 3, 4))
            segments[-1] = segments[-1].reshape(2, num_kv_heads, prefix_len, head_dim)
        for j, p in enumerate(own[i]):
            n = min(tpb, cached[i] - j * tpb)
            if n > 0:
                segments.append(poolf[p, :, :, :n, :])
        kv = torch.cat(segments, dim=2)
        keys = torch.cat([kv[0], k[b].float()[:, None, :]], dim=1)
        values = torch.cat([kv[1], v[b].float()[:, None, :]], dim=1)
        qh = q[b].float().view(num_kv_heads, gs, head_dim)
        probs = torch.softmax(torch.einsum("hgd,htd->hgt", qh, keys) * scale, dim=-1)
        expected = torch.einsum("hgt,htd->hgd", probs, values).reshape(num_heads, head_dim)
        torch.testing.assert_close(out[b].float(), expected, atol=3e-2, rtol=3e-2)
        # The current token's K/V reached its cache slot.
        slot_page, slot = int(write_page[b]), int(write_slot[b])
        torch.testing.assert_close(pool[slot_page, 0, :, slot, :], k[b])
        torch.testing.assert_close(pool[slot_page, 1, :, slot, :], v[b])


def _fused_spec_case(
    num_heads,
    num_kv_heads,
    head_dim,
    q_len,
    shuffled,
    placeholder_rows,
    seed,
    overshoot,
    peaked=False,
    massive=False,
):
    """Tables, tensors and dense fp32 reference of one speculative fused-kernel step.

    ``cached[r]`` private tokens precede the step's ``q_len`` tokens. With
    ``overshoot`` the host page table of every row is one page longer than the
    device length needs (tail last-page length <= 0 for rows whose tokens end a
    page), as under the overlap scheduler.
    """
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(seed)
    tpb, groups, per_group, prefix_pages = 32, 3, 5, 4
    sharing_rows = groups * per_group
    rows = sharing_rows + placeholder_rows
    # Tail lengths hitting page boundaries (-q_len, -1, 0, +1 around multiples of tpb) and random ones.
    special = [tpb - q_len, tpb - 1, tpb, tpb + 1, 2 * tpb - q_len, 0, 1]
    cached = [
        special[r % len(special)] if r % 2 == 0 else int(torch.randint(0, 90, (1,), generator=gen))
        for r in range(rows)
    ]
    cached = [max(c, 0) if r < sharing_rows else max(c, 0) + 70 for r, c in enumerate(cached)]
    need = [(c + q_len + tpb - 1) // tpb for c in cached]
    blocks = [n + (1 if overshoot else 0) for n in need]
    num_pages = groups * prefix_pages + 1 + sum(blocks)
    pool = (0.7 * torch.randn(num_pages, 2, num_kv_heads, tpb, head_dim, device=device)).bfloat16()
    ntok = rows * q_len
    q = torch.randn(ntok, num_heads, head_dim, device=device).bfloat16()
    k = torch.randn(ntok, num_kv_heads, head_dim, device=device).bfloat16()
    v = torch.randn(ntok, num_kv_heads, head_dim, device=device).bfloat16()
    perm = torch.randperm(rows, generator=gen) if shuffled else torch.arange(rows)
    placeholder_page = groups * prefix_pages
    page, own = placeholder_page + 1, []
    for b in blocks:
        own.append(list(range(page, page + b)))
        page += b
    if massive:
        # Massive-activation magnitudes of real models (Qwen2.5-7B layer 27: |k| ~ 420, |q| ~ 66, scaled logits
        # of -3500..-6800 for some heads): keys of one sign, queries of either sign per head, so odd heads have
        # every logit hugely negative and even heads random-signed. A kernel that clamps the running maximum
        # (FlashInfer FA2 starts it at -5e4 and uses 5e4 for infinity) returns zeros for such heads.
        kmag = (pool[:, 0].float().abs() * 600.0).bfloat16()
        pool[:, 0] = kmag
        sign = torch.where(torch.arange(num_heads, device=device) % 2 == 1, -1.0, 1.0)[
            None, :, None
        ]
        q = (q.float().abs() * 80.0 * sign + 0.0 * q.float()).bfloat16()
        q[:, ::2] = (torch.randn_like(q[:, ::2].float()) * 80.0).bfloat16()
        k = (k.float().abs() * 600.0).bfloat16()
    if peaked:
        # Attention dominated by the newest keys (as in a real model, unlike flat random
        # attention over a thousand keys): every query is a large multiple of one direction per
        # head, and the key of each row's last cached token and of its first step token point
        # along it. A wrong tail length (a dropped newest token, an unwritten or stale slot read
        # instead) then changes the output by O(1), not by the 1/length of flat attention.
        gs = num_heads // num_kv_heads
        u = torch.randn(num_heads, head_dim, device=device)
        u = u / u.norm(dim=-1, keepdim=True) * math.sqrt(head_dim)
        q = (6.0 * u).bfloat16()[None].expand(ntok, -1, -1).contiguous()
        ukv = u.view(num_kv_heads, gs, head_dim)[:, 0]
        for r in range(rows):
            if cached[r] > 0:
                pos = cached[r] - 1
                pool[own[r][pos // tpb], 0, :, pos % tpb] = (2.0 * ukv).bfloat16()
        b_of = perm.tolist()
        for r in range(rows):
            k[b_of[r] * q_len] = (2.0 * ukv).bfloat16()
    # Device-derived last-page length: true tail tokens minus the pages before the last host page.
    last = [c + q_len - (len(o) - 1) * tpb for c, o in zip(cached, own)]
    tail_indptr = [0]
    for o in own:
        tail_indptr.append(tail_indptr[-1] + len(o))
    # Query-token ranges of the groups (q_len tokens per row), as the FlashInfer prefix call takes them.
    qo = [g * per_group * q_len for g in range(groups + 1)] + [
        (rows if placeholder_rows else groups * per_group) * q_len
    ]
    tables = dict(
        perm=perm.to(device, torch.int64),
        qo=qo,
        prefix_indptr=[g * prefix_pages for g in range(groups + 1)] + [groups * prefix_pages + 1],
        prefix_indices=list(range(groups * prefix_pages)) + [placeholder_page],
        prefix_last=[tpb] * groups + [1],
        empty=[False] * sharing_rows + [True] * placeholder_rows,
        tail_indptr=tail_indptr,
        tail_indices=[p for o in own for p in o],
        last=last,
    )
    # Dense fp32 reference in float64 on the pre-step pool.
    pf, gs = pool.double(), num_heads // num_kv_heads
    ref = torch.zeros(ntok, num_heads, head_dim, dtype=torch.float64, device=device)
    for i in range(rows):
        b = int(perm[i])
        segs = []
        if i < sharing_rows:
            g = i // per_group
            segs.append(
                pf[g * prefix_pages : (g + 1) * prefix_pages]
                .permute(1, 2, 0, 3, 4)
                .reshape(2, num_kv_heads, -1, head_dim)
            )
        for j, p in enumerate(own[i]):
            n = min(tpb, cached[i] - j * tpb)
            if n > 0:
                segs.append(pf[p, :, :, :n, :])
        kv = torch.cat(segs, dim=2)
        for t in range(q_len):
            keys = torch.cat([kv[0], k[b * q_len : b * q_len + t + 1].double().permute(1, 0, 2)], 1)
            vals = torch.cat([kv[1], v[b * q_len : b * q_len + t + 1].double().permute(1, 0, 2)], 1)
            qh = q[b * q_len + t].double().view(num_kv_heads, gs, head_dim)
            probs = torch.softmax(
                torch.einsum("hgd,htd->hgt", qh, keys) / math.sqrt(head_dim), dim=-1
            )
            ref[b * q_len + t] = torch.einsum("hgt,htd->hgd", probs, vals).reshape(
                num_heads, head_dim
            )
    return dict(
        pool=pool, q=q, k=k, v=v, ref=ref, tables=tables, rows=rows, cached=cached, own=own,
        tpb=tpb, groups=groups, per_group=per_group, ntok=ntok, sharing_rows=sharing_rows,
    )  # fmt: skip


def _make_ctx(case, num_heads, head_dim, q_len):
    """Context-kernel prefix tables of a fused-spec case (skips when the kernel is unavailable)."""
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix import _ctx_tables
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fused import (
        CtxPrefix,
        ctx_prefix_supported,
    )

    if not ctx_prefix_supported(head_dim, torch.bfloat16):
        pytest.skip("TRTLLM-Gen context kernel needs SM100/SM103")
    t = case["tables"]
    groups = case["groups"] + 1
    fake = SimpleNamespace(
        num_groups=groups,
        qo_indptr=torch.tensor(t["qo"], dtype=torch.int32),
        prefix_indptr=torch.tensor(t["prefix_indptr"], dtype=torch.int32),
        prefix_indices=torch.tensor(t["prefix_indices"], dtype=torch.int32),
        prefix_last_page_len=torch.tensor(t["prefix_last"], dtype=torch.int32),
    )
    requests = CtxPrefix.num_requests_for(groups + 2, case["ntok"], q_len)
    max_pages = 8
    ctx = CtxPrefix(
        case["ntok"], num_heads, head_dim, q_len, requests, max_pages, 16384, torch.bfloat16,
        torch.device("cuda"), case["tpb"],
    )  # fmt: skip
    for dst, host in zip(
        ctx.table_arrays(), _ctx_tables(fake, case["tpb"], ctx.chunk_tokens, requests, max_pages)
    ):
        dst.copy_(torch.from_numpy(host))
    return ctx


def _run_fused_spec(
    case, num_heads, head_dim, q_len, pool, q, k, v, out, workspace, group_grid=None, ctx=None
):
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fused import (
        fused_sampling_attention,
    )

    device = torch.device("cuda")
    t = case["tables"]

    if "dev" not in case:  # device copies made once, outside any graph capture

        def i32(x):
            return torch.tensor(x, dtype=torch.int32, device=device)

        case["dev"] = {
            name: i32(t[name])
            for name in ("qo", "prefix_indptr", "prefix_indices", "prefix_last", "tail_indptr",
                         "tail_indices", "last")
        }  # fmt: skip
        case["dev"]["empty"] = torch.tensor(t["empty"], device=device)
        case["dev"]["zeros"] = torch.zeros(case["rows"], dtype=torch.int64, device=device)
    d = case["dev"]
    fused_sampling_attention(
        pool, q, k, v, out,
        perm=t["perm"], group_qo_indptr=d["qo"], prefix_indptr=d["prefix_indptr"],
        prefix_indices=d["prefix_indices"], prefix_last_page_len=d["prefix_last"],
        prefix_empty=d["empty"],
        tail_indptr=d["tail_indptr"], tail_indices=d["tail_indices"],
        tail_last_page_len=d["last"], write_page=d["zeros"], write_slot=d["zeros"],
        sm_scale=1.0 / math.sqrt(head_dim), num_groups=case["groups"] + 1,
        max_group_rows=case["per_group"], workspace=workspace, q_len=q_len,
        prefix_splits=2, tail_splits=1, group_grid=group_grid, ctx=ctx,
    )  # fmt: skip


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,q_len,shuffled,placeholder_rows,overshoot,group_grid",
    [
        (20, 20, 128, 9, True, 2, True, 2),
        (16, 8, 128, 2, False, 0, False, None),
        (12, 3, 128, 5, True, 2, True, 2),
    ],
)
def test_fused_spec_attention_matches_dense_and_writes_kv(
    num_heads, num_kv_heads, head_dim, q_len, shuffled, placeholder_rows, overshoot, group_grid
):
    """q_len tokens per row: causal among themselves, attending the group's shared pages and
    the row's private pages; their K/V land at tail position cached + j (page crossings,
    host page tables a page ahead of the device length, row permutation, placeholder rows)."""
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fused import (
        FusedWorkspace,
    )

    device = torch.device("cuda")
    for seed in range(2):
        case = _fused_spec_case(
            num_heads, num_kv_heads, head_dim, q_len, shuffled, placeholder_rows, seed, overshoot
        )
        pool = case["pool"]
        before = pool.clone()
        out = torch.empty_like(case["q"])
        ws = FusedWorkspace(case["ntok"], num_heads, head_dim, 3, device)
        _run_fused_spec(
            case,
            num_heads,
            head_dim,
            q_len,
            pool,
            case["q"],
            case["k"],
            case["v"],
            out,
            ws,
            group_grid=group_grid,
        )
        torch.cuda.synchronize()
        err = (out.double() - case["ref"]).abs().amax() / case["ref"].abs().amax()
        assert float(err) < 2e-2, f"seed {seed}: rel err {float(err)}"
        # K/V of the step's tokens at cached + j; every other slot unchanged.
        expect = before.clone()
        for i in range(case["rows"]):
            b = int(case["tables"]["perm"][i])
            for j in range(q_len):
                pos = case["cached"][i] + j
                page = case["own"][i][pos // case["tpb"]]
                expect[page, 0, :, pos % case["tpb"]] = case["k"][b * q_len + j]
                expect[page, 1, :, pos % case["tpb"]] = case["v"][b * q_len + j]
        assert torch.equal(pool, expect)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,q_len,prefix_kernel",
    [(28, 4, 128, 2, "triton"), (28, 4, 128, 2, "ctx")],
)
def test_fused_attention_with_massive_activations(
    num_heads, num_kv_heads, head_dim, q_len, prefix_kernel
):
    """Keys ~600x, queries ~80x the usual scale, odd heads with every logit hugely negative, against a
    float64 reference: no zero heads, output within bf16 rounding of the largest value of each head, for
    the Triton prefix, the TRTLLM-Gen context-kernel prefix (its LSE feeds the merge) and the Triton tail."""
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fused import (
        FusedWorkspace,
    )

    device = torch.device("cuda")
    for seed in range(2):
        torch.manual_seed(seed)
        case = _fused_spec_case(
            num_heads, num_kv_heads, head_dim, q_len, True, 2, seed, True, massive=True
        )
        pool = case["pool"]
        out = torch.empty_like(case["q"])
        ws = FusedWorkspace(case["ntok"], num_heads, head_dim, 3, device)
        ctx = _make_ctx(case, num_heads, head_dim, q_len) if prefix_kernel == "ctx" else None
        _run_fused_spec(
            case, num_heads, head_dim, q_len, pool, case["q"], case["k"], case["v"], out, ws,
            ctx=ctx,
        )  # fmt: skip
        torch.cuda.synchronize()
        ref = case["ref"]
        head_max = ref.abs().amax(-1)  # [tokens, heads]
        err = (out.double() - ref).abs().amax(-1)
        assert not bool(((out.abs().amax(-1) == 0) & (head_max > 1e-3)).any()), "zero head"
        assert (
            float((err / head_max.clamp(min=1e-3)).max()) < 2.5e-2
        )  # ~1 bf16 ulp of the head's largest value (the context kernel rounds its output once more)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("q_len", [9])
def test_fused_spec_attention_graph_replay_is_bit_exact(q_len):
    """One CUDA graph of the fused speculative kernel, replayed on new inputs, equals eager
    calls on the same buffers bit for bit."""
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fused import (
        FusedWorkspace,
    )

    device = torch.device("cuda")
    num_heads, num_kv_heads, head_dim = 20, 20, 128
    case = _fused_spec_case(num_heads, num_kv_heads, head_dim, q_len, True, 2, 7, True)
    FusedWorkspace.shared(8, num_heads, head_dim, 3, device)  # a smaller arena first
    ws = FusedWorkspace.shared(case["ntok"], num_heads, head_dim, 3, device)
    q, k, v = case["q"].clone(), case["k"].clone(), case["v"].clone()
    pool = case["pool"].clone()
    out_graph = torch.zeros_like(q)
    args = (case, num_heads, head_dim, q_len)
    _run_fused_spec(*args, pool, q, k, v, out_graph, ws)  # warm up / compile
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_fused_spec(*args, pool, q, k, v, out_graph, ws)
    gen = torch.Generator(device=device).manual_seed(11)
    for _ in range(3):
        q.copy_(torch.randn(q.shape, generator=gen, device=device).bfloat16())
        k.copy_(torch.randn(k.shape, generator=gen, device=device).bfloat16())
        v.copy_(torch.randn(v.shape, generator=gen, device=device).bfloat16())
        eager_pool = pool.clone()
        eager_out = torch.zeros_like(q)
        _run_fused_spec(*args, eager_pool, q, k, v, eager_out, ws)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_graph, eager_out)
        assert torch.equal(pool, eager_pool)


def _eagle3_paths():
    """(target, draft) checkpoints of the EAGLE-3 e2e test, from the environment."""
    target = os.environ.get("TLLM_TEST_EAGLE3_TARGET")
    draft = os.environ.get("TLLM_TEST_EAGLE3_DRAFT")
    if target and draft and os.path.isdir(target) and os.path.isdir(draft):
        return target, draft
    return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(
    _eagle3_paths() is None,
    reason="set TLLM_TEST_EAGLE3_TARGET / TLLM_TEST_EAGLE3_DRAFT (e.g. Qwen3-1.7B and its EAGLE-3 head)",
)
@pytest.mark.threadleak(enabled=False)
def test_eagle3_n_sampling_with_fused_library_keeps_acceptance_e2e():
    """EAGLE-3 (3 draft tokens) with n-sample requests, CUDA graphs on: with the library
    forced open (fused spec kernel in the captured graphs) every child produces its tokens and the
    mean accepted draft length stays within 5% of the run without the library."""
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig

    target, draft = _eagle3_paths()
    prompts = [[1024 + ((29 * i + 7 * p) % 50000) for p in range(900 - 100 * i)] for i in range(2)]
    n, max_tokens = 8, 96

    def run(env):
        previous = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            llm = LLM(
                model=target,
                max_batch_size=len(prompts) * n,
                max_num_tokens=8192,
                max_seq_len=1100,
                speculative_config=Eagle3DecodingConfig(max_draft_len=3, speculative_model=draft),
                kv_cache_config=KvCacheConfig(
                    enable_block_reuse=False,
                    use_kv_cache_manager_v2=True,
                    free_gpu_memory_fraction=0.3,
                ),
                cuda_graph_config=CudaGraphConfig(max_batch_size=len(prompts) * n),
                skip_tokenizer_init=True,
                enable_iter_perf_stats=True,
            )
            with llm:
                sampling = SamplingParams(
                    n=n, temperature=1.0, max_tokens=max_tokens, ignore_eos=True, end_id=-1
                )
                outputs = llm.generate(prompts, sampling_params=sampling)
                stats = llm.get_stats(2) or []
            accepted = requests = 0
            for stat in stats:
                spec = stat.get("specDecodingStats") if isinstance(stat, dict) else None
                if spec and spec.get("numRequestsWithDraftTokens", 0) > 0:
                    accepted += spec["numAcceptedTokens"]
                    requests += spec["numRequestsWithDraftTokens"]
            assert all(len(c.token_ids) == max_tokens for o in outputs for c in o.outputs)
            return (accepted + requests) / requests if requests else None
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    off = run({})
    on = run(
        {
            "TLLM_FMHA_LIBS": "+sampling_shared_prefix",
            "TLLM_SAMPLING_SHARED_PREFIX_MIN_SHARED_TOKENS": "0",
            "TLLM_SAMPLING_SHARED_PREFIX_SPEC_MIN_SHARED_TOKENS": "0",
            "TLLM_SAMPLING_SHARED_PREFIX_MIN_GROUP": "2",
        }
    )
    if off is not None and on is not None:
        assert abs(on - off) <= 0.05 * off, (on, off)
