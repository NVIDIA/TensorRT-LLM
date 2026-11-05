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
from types import SimpleNamespace

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention.backends.fmha.beam_shared_prefix import (
    LIB_NAME,
    merge_prefix_and_tail,
    metadata_eligible,
    plan_step,
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


class _ManagerV2:
    """The KV cache manager V2 surface the library reads, with the page tables of beam 0."""

    def __init__(self, beam0_pages, *, page_scale=2, num_pages=256, **overrides):
        self.beam0_pages = beam0_pages  # request id -> pages of the layer's get_buffers view
        self.page_scale = page_scale
        self.num_pages = num_pages
        self.layer_offsets = {0: 0, 1: 1}
        self.layer_to_pool_mapping_dict = {0: 0, 1: 0}
        self.index_mapper = object()
        self.kv_factor = 2
        self.tokens_per_block = 4
        self.max_beam_width = 4
        self.__dict__.update(overrides)

    def get_buffers(self, layer_idx, kv_layout="NHD"):
        return torch.empty(self.num_pages, 2, 1, self.tokens_per_block, 8)

    def get_layer_page_index_scale(self, layer_idx):
        return self.page_scale

    def get_batch_cache_indices(self, request_ids, layer_idx=None, num_blocks_per_seq=None):
        return [self.beam0_pages[rid][:n] for rid, n in zip(request_ids, num_blocks_per_seq)]


def _beam_meta(manager, *, beam_width=4, request_ids=(7, 9), kv_lens=(11, 7),
               prompt_lens=(10, 6)):  # fmt: skip
    """A generation-only beam-search batch: one row per beam, request-major."""
    rows = beam_width * len(request_ids)
    expand = lambda values: torch.tensor(values).repeat_interleave(beam_width)  # noqa: E731
    return SimpleNamespace(
        beam_width=beam_width,
        num_generations=rows,
        num_contexts=0,
        is_cross=False,
        enable_helix=False,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        cache_indirection=torch.zeros(len(request_ids), beam_width, 16, dtype=torch.int32),
        kv_cache_block_offsets=torch.zeros(1, rows, 2, 8, dtype=torch.int32),
        kv_cache_params=SimpleNamespace(use_cache=True, num_extra_kv_tokens=0),
        kv_cache_manager=manager,
        fmha_plan_caches={},
        is_cuda_graph=False,
        request_ids=list(request_ids),
        kv_lens_runtime=expand(kv_lens),
        prompt_lens_cpu_runtime=expand(prompt_lens),
        tokens_per_block=manager.tokens_per_block,
    )


def test_only_kv_manager_v2_with_one_table_per_pool_is_eligible():
    """The library reads the V2 block table: a V1 manager (no index mapper) and a V2 manager with
    per-layer page tables (indexed by local layer, not by pool) are declined."""
    assert metadata_eligible(_beam_meta(_ManagerV2({})))
    v1 = _ManagerV2({})
    del v1.index_mapper
    assert not metadata_eligible(_beam_meta(v1))
    assert not metadata_eligible(_beam_meta(_ManagerV2({}, _use_per_layer_page_tables=True)))
    assert not metadata_eligible(_beam_meta(_ManagerV2({}), beam_width=1))


def test_plan_step_reads_beam0_prompt_pages_from_kv_manager_v2(monkeypatch):
    """The prefix page table holds beam 0's prompt pages of every request (from the V2 manager, in
    batch order), the last page's fill, ``beam_width`` queries per request and the tail length."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"+{LIB_NAME}")
    manager = _ManagerV2({7: [40, 41, 42, 43], 9: [50, 51, 52]})
    meta = _beam_meta(manager)
    plan_step(meta)
    plan = meta.fmha_plan_caches[LIB_NAME].plan
    # Request 7: 10 prompt tokens in 4-token blocks -> 3 pages, 2 tokens on the last one.
    # Request 9: 6 prompt tokens -> 2 pages, 2 on the last.
    assert plan.paged_kv_indices.tolist() == [40, 41, 42, 50, 51]
    assert plan.paged_kv_indptr.tolist() == [0, 3, 5]
    assert plan.paged_kv_last_page_len.tolist() == [2, 2]
    assert plan.qo_indptr.tolist() == [0, 4, 8]
    assert plan.tail_len == 0  # every beam holds only the current generated token

    monkeypatch.setenv("TLLM_FMHA_LIBS", "-" + LIB_NAME)
    off = _beam_meta(manager)
    plan_step(off)
    assert off.fmha_plan_caches == {}


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
