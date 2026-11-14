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

import functools
import gc
import math
import os
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.attention.backends.fmha.beam_shared_prefix import (
    LIB_NAME,
    BeamSharedPrefixFmha,
    _BeamStepState,
    _Config,
    _create_wrapper,
    _float_workspace,
    _get_float_workspace,
    _HostPlan,
    _plan_wrapper,
    _PlanKey,
    _step_indices,
    _StepIndices,
    merge_prefix_and_tail,
    metadata_eligible,
    plan_step,
)
from tensorrt_llm._torch.attention.backends.fmha.beam_shared_prefix_kernels import (
    fused_tail_and_merge,
)
from tensorrt_llm._torch.attention.backends.fmha.shared_prefix_fa4 import (
    PrefixGroups,
    prefix_request_tables,
)
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm.functional import PositionEmbeddingType

_LOG2E = 1.4426950408889634
needs_flashinfer = pytest.mark.skipif(not IS_FLASHINFER_AVAILABLE, reason="needs FlashInfer")


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


@pytest.mark.parametrize("tail_len, with_mask, group", [(4, True, 2), (1, False, 4), (3, False, 1)])
def test_merge_prefix_and_tail_matches_dense_attention(tail_len, with_mask, group):
    """Merge prefix and tail matches dense attention."""
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
        """Record the fake manager's page tables."""
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
        """A pool of the fake manager's size (only its page count is read)."""
        return torch.empty(self.num_pages, 2, 1, self.tokens_per_block, 8)

    def get_layer_page_index_scale(self, layer_idx):
        """Page-index scale of the fake pool."""
        return self.page_scale

    def get_batch_cache_indices(self, request_ids, layer_idx=None, num_blocks_per_seq=None):
        """Pages of each request's prompt."""
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


# Group size only changes the tile padding (1, 2, 4, 8 share one code path; 3 and 6 need padding),
# head size only the tile, tail length 0/1/3 are distinct paths, dtype only the casts, and the FA4
# layout only the LSE strides and scale.
_FUSED_CASES = [
    (1, 3, 128, torch.bfloat16, False),
    (4, 3, 128, torch.bfloat16, False),
    (8, 1, 64, torch.bfloat16, False),
    (4, 0, 128, torch.bfloat16, False),
    (2, 3, 64, torch.float16, False),
    (8, 0, 128, torch.float16, False),
    (3, 3, 128, torch.bfloat16, False),
    (6, 1, 64, torch.bfloat16, False),
    (4, 3, 256, torch.bfloat16, False),
    (2, 3, 128, torch.bfloat16, True),
    (4, 0, 64, torch.bfloat16, True),
    (8, 1, 128, torch.bfloat16, True),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("group, tail_len, head_dim, dtype, fa4_layout", _FUSED_CASES)
def test_fused_tail_and_merge_matches_unfused_reference(
    group, tail_len, head_dim, dtype, fa4_layout
):
    """The fused kernel reproduces the unfused tail attention and merge, and the KV write.

    ``fa4_layout`` hands it the prefix state the way ``Fa4Prefix`` leaves it: natural-log LSE
    [heads, capacity] and an output buffer larger than the batch.
    """
    torch.manual_seed(0)
    device = "cuda"
    num_kv_heads, tokens_per_block, rows = 2, 16, 24
    num_heads = num_kv_heads * group
    num_pages = 3 * rows
    sm_scale = head_dim**-0.5
    pool = torch.randn(num_pages, 2, num_kv_heads, tokens_per_block, head_dim, device=device).to(
        dtype
    )
    # q, k and v are strided slices of one packed QKV tensor, as in the model.
    qkv = torch.randn(rows, (num_heads + 2 * num_kv_heads) * head_dim, device=device).to(dtype)
    q = qkv[:, : num_heads * head_dim].view(rows, num_heads, head_dim)
    k = qkv[:, num_heads * head_dim :][:, : num_kv_heads * head_dim].view(
        rows, num_kv_heads, head_dim
    )
    v = qkv[:, (num_heads + num_kv_heads) * head_dim :].view(rows, num_kv_heads, head_dim)
    prefix_out = torch.randn(rows, num_heads, head_dim, device=device).to(dtype)
    prefix_lse = torch.randn(rows, num_heads, device=device) * 2
    prefix_empty = torch.zeros(rows, dtype=torch.bool, device=device)
    prefix_empty[::7] = True

    # Current tokens go to their own pages; past positions live in a disjoint range.
    write_page = torch.arange(rows, device=device, dtype=torch.int64)
    write_slot = torch.randint(0, tokens_per_block, (rows,), device=device, dtype=torch.int64)
    tail_page = tail_slot = tail_valid = None
    if tail_len:
        tail_page = torch.randint(rows, 3 * rows, (rows, tail_len), device=device)
        tail_slot = torch.randint(0, tokens_per_block, (rows, tail_len), device=device)
        tail_valid = torch.rand(rows, tail_len, device=device) > 0.3
    indices = _StepIndices(
        step=0,
        write_page=write_page,
        write_slot=write_slot,
        tail_page=tail_page,
        tail_slot=tail_slot,
        tail_valid=tail_valid,
        prefix_empty=prefix_empty,
        captured=False,
    )

    if tail_len:
        k_all = torch.cat([pool[tail_page, 0, :, tail_slot, :], k.unsqueeze(1)], dim=1)
        v_all = torch.cat([pool[tail_page, 1, :, tail_slot, :], v.unsqueeze(1)], dim=1)
        valid = torch.cat([tail_valid, torch.ones(rows, 1, dtype=torch.bool, device=device)], 1)
    else:
        k_all, v_all, valid = k.unsqueeze(1), v.unsqueeze(1), None
    expected = merge_prefix_and_tail(
        q,
        k_all,
        v_all,
        valid,
        prefix_out,
        prefix_lse,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        prefix_empty=prefix_empty,
    )

    output = torch.empty(rows, num_heads, head_dim, device=device, dtype=dtype)
    lse_scale = 1.0
    if fa4_layout:
        capacity = rows + 8
        lse_buffer = torch.randn(num_heads, capacity, device=device)
        lse_buffer[:, :rows] = (prefix_lse / _LOG2E).t()
        out_buffer = torch.randn(capacity, num_heads, head_dim, device=device).to(dtype)
        out_buffer[:rows] = prefix_out
        prefix_out_arg, prefix_lse_arg, lse_scale = out_buffer[:rows], lse_buffer, _LOG2E
    else:
        prefix_out_arg, prefix_lse_arg = prefix_out, prefix_lse
    fused_tail_and_merge(
        q,
        k,
        v,
        pool,
        indices,
        prefix_out_arg,
        prefix_lse_arg,
        output,
        tail_len=tail_len,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
        lse_scale=lse_scale,
        lse_heads_first=fa4_layout,
    )
    torch.testing.assert_close(output.float(), expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(pool[write_page, 0, :, write_slot, :], k, atol=0, rtol=0)
    torch.testing.assert_close(pool[write_page, 1, :, write_slot, :], v, atol=0, rtol=0)


@needs_flashinfer
def test_only_kv_manager_v2_with_one_table_per_pool_is_eligible():
    """The library reads the V2 block table: a V1 manager (no index mapper) and a V2 manager with
    per-layer page tables (indexed by local layer, not by pool) are declined."""
    assert metadata_eligible(_beam_meta(_ManagerV2({})))
    v1 = _ManagerV2({})
    del v1.index_mapper
    assert not metadata_eligible(_beam_meta(v1))
    assert not metadata_eligible(_beam_meta(_ManagerV2({}, _use_per_layer_page_tables=True)))
    assert not metadata_eligible(_beam_meta(_ManagerV2({}), beam_width=1))


@needs_flashinfer
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


@needs_flashinfer
def test_plan_step_with_prompts_that_fill_their_last_page_exactly(monkeypatch):
    """A prompt that is a multiple of the block size has a full last page, not an empty one."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"+{LIB_NAME}")
    manager = _ManagerV2({7: [40, 41, 42], 9: [50, 51]})
    meta = _beam_meta(manager, kv_lens=(9, 5), prompt_lens=(8, 4))
    plan_step(meta)
    plan = meta.fmha_plan_caches[LIB_NAME].plan
    assert plan.paged_kv_indices.tolist() == [40, 41, 50]
    assert plan.paged_kv_last_page_len.tolist() == [4, 4]


@needs_flashinfer
def test_plan_step_under_cuda_graphs_fixes_the_tail_and_rejects_a_longer_one(monkeypatch):
    """Under CUDA graphs the tail length is fixed at ``MAX_TAIL`` when the graph is captured, so a
    beam with more generated positions than that fails loudly instead of being truncated."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"+{LIB_NAME}")
    manager = _ManagerV2({7: [40, 41, 42, 43], 9: [50, 51, 52]})
    monkeypatch.setenv("TLLM_BEAM_SHARED_PREFIX_MAX_TAIL", "6")
    meta = _beam_meta(manager)
    meta.is_cuda_graph = True
    plan_step(meta)
    assert meta.fmha_plan_caches[LIB_NAME].plan.tail_len == 6
    monkeypatch.setenv("TLLM_BEAM_SHARED_PREFIX_MAX_TAIL", "2")
    longer = _beam_meta(manager, kv_lens=(15, 7))  # request 7 carries 4 generated positions
    longer.is_cuda_graph = True
    with pytest.raises(RuntimeError, match="CUDA graphs were captured"):
        plan_step(longer)


def test_step_indices_gather_the_tail_through_cache_indirection():
    """Past generated positions are read from the block of the beam that produced them (per the
    sampler's indirection), across a block boundary, behind a context row, with page stride 2."""
    num_ctx, beam_width, tokens_per_block, page_stride = 1, 2, 4, 2
    prompt_lens, kv_lens = (6, 5), (10, 8)  # tails of 3 and 2 generated positions; tail_len 3
    rows = beam_width * len(prompt_lens)
    offsets = torch.zeros(1, num_ctx + rows, 2, 6, dtype=torch.int32)
    for row in range(num_ctx + rows):
        for block in range(6):
            offsets[0, row, 0, block] = page_stride * (10 * row + block)  # page 10*row + block
    indirection = torch.zeros(len(prompt_lens), beam_width, 16, dtype=torch.int32)
    indirection[:, 0, :] = 1  # beam 0 reads positions written by beam 1, and the reverse
    meta = SimpleNamespace(
        num_contexts=num_ctx,
        tokens_per_block=tokens_per_block,
        kv_cache_block_offsets=offsets,
        kv_lens_cuda_runtime=torch.tensor([5, *[v for v in kv_lens for _ in range(beam_width)]]),
        prompt_lens_cuda_runtime=torch.tensor(
            [5, *[p for p in prompt_lens for _ in range(beam_width)]]
        ),
        cache_indirection=indirection,
    )
    plan = _HostPlan(
        step=1,
        num_requests=len(prompt_lens),
        beam_width=beam_width,
        qo_indptr=torch.zeros(3, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(3, dtype=torch.int32),
        paged_kv_indices=torch.zeros(1, dtype=torch.int32),
        paged_kv_last_page_len=torch.zeros(2, dtype=torch.int32),
        tail_len=3,
    )
    state = _BeamStepState(
        is_cuda_graph=False,
        pool_index=0,
        page_scale=page_stride,
        page_stride=page_stride,
        config=_Config.from_env(),
        num_pages=1000,
        plan=plan,
    )
    got = _step_indices(state, meta, 1000)

    for row in range(rows):
        request, beam = divmod(row, beam_width)
        current = kv_lens[request] - 1
        assert got.write_page[row] == 10 * (num_ctx + row) + current // tokens_per_block
        assert got.write_slot[row] == current % tokens_per_block
        for i in range(3):
            position = prompt_lens[request] + i
            source_row = num_ctx + request * beam_width + (1 - beam)
            assert got.tail_valid[row, i] == (position < current)
            if position < current:
                assert got.tail_page[row, i] == 10 * source_row + position // tokens_per_block
                assert got.tail_slot[row, i] == position % tokens_per_block


@needs_flashinfer
def test_plan_step_drops_the_previous_plan_when_a_step_is_not_planned(monkeypatch):
    """A step that declines planning must not leave the last step's page table to be reused."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"+{LIB_NAME}")
    manager = _ManagerV2({7: [40, 41, 42, 43], 9: [50, 51, 52]})
    meta = _beam_meta(manager)
    plan_step(meta)
    state = meta.fmha_plan_caches[LIB_NAME]
    assert state.plan is not None
    declined = _beam_meta(manager, beam_width=1, request_ids=(7,), kv_lens=(11,), prompt_lens=(10,))
    declined.fmha_plan_caches = meta.fmha_plan_caches
    plan_step(declined)
    assert state.plan is None


def test_prefix_request_tables_chunk_skip_empty_groups_and_size_the_last_page():
    """Prefix request tables chunk skip empty groups and size the last page."""
    i32 = lambda *v: np.array(v, dtype=np.int32)  # noqa: E731
    # Group 0: 200 queries over 3 pages (last holds 1 token); group 1: empty; group 2: 4 queries.
    groups = PrefixGroups(
        num_groups=3,
        qo_indptr=i32(0, 200, 200, 204),
        prefix_indptr=i32(0, 3, 3, 5),
        prefix_indices=i32(10, 11, 12, 20, 21),
        prefix_last_page_len=i32(1, 1, 16),
    )
    tables, seq_lens, cum_q, cum_kv = prefix_request_tables(
        groups, tokens_per_block=32, chunk_tokens=128, num_requests=5, max_pages=4
    )
    assert seq_lens.tolist() == [65, 65, 48, 0, 0]  # (count - 1) * 32 + last page
    assert tables[0, :3].tolist() == tables[1, :3].tolist() == [10, 11, 12]
    assert tables[2, :2].tolist() == [20, 21]
    assert cum_q.tolist() == [0, 128, 200, 204, 204, 204]
    assert cum_kv.tolist() == [0, 65, 130, 178, 178, 178]
    with pytest.raises(RuntimeError, match="more prefix requests"):
        prefix_request_tables(groups, 32, 128, num_requests=2, max_pages=4)


def test_float_workspace_is_owned_per_kv_cache_manager_and_released_with_it():
    """Float workspace is owned per kv cache manager and released with it."""

    class Manager:
        """A stand-in KV cache manager."""

        pass

    cpu = torch.device("cpu")
    first, second = Manager(), Manager()
    a = _get_float_workspace(first, cpu, 1024)
    assert _get_float_workspace(first, cpu, 1024) is a
    assert _get_float_workspace(second, cpu, 1024) is not a
    second_ref = weakref.ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert first in _float_workspace


@needs_flashinfer
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_replanning_a_cuda_graph_wrapper_does_not_wait_for_the_gpu():
    """Under CUDA graphs every step re-plans the wrapper; that must not synchronize the stream
    (it would serialize the host with the previous step and defeat the overlap scheduler)."""

    class Manager:
        """A stand-in KV cache manager."""

        max_blocks_per_seq = 8
        max_batch_size = 2
        max_beam_width = 4

    device = torch.device("cuda")
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.zeros(1, device=device),
        kv_cache_manager=Manager(),
        tokens_per_block=16,
    )
    plan = _HostPlan(
        step=1,
        num_requests=2,
        beam_width=4,
        qo_indptr=torch.tensor([0, 4, 8], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 3, 5], dtype=torch.int32),
        paged_kv_indices=torch.tensor([3, 4, 5, 8, 9], dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([5, 16], dtype=torch.int32),
        tail_len=8,
    )
    state = SimpleNamespace(
        is_cuda_graph=True,
        plan=plan,
        planned_step={},
        config=_Config(backend="fa2", fused_tail=True, check=False, max_tail=8),
    )
    key = _PlanKey(8, 2, 64, torch.bfloat16, 0.125)
    wrapper = _create_wrapper(state, key, metadata)
    _plan_wrapper(state, key, wrapper, metadata)  # warm up (builds the plan module)
    torch.cuda.synchronize()

    torch.cuda._sleep(2_000_000_000)  # keep the stream busy for about a second
    busy = torch.cuda.Event()
    busy.record()
    _plan_wrapper(state, key, wrapper, metadata)
    returned_while_busy = not busy.query()
    torch.cuda.synchronize()
    assert returned_while_busy, "re-planning waited for the GPU stream"


def test_run_generation_without_a_step_plan_runs_the_regular_kernel():
    """The manager caches a selection by batch shape, so an unplanned batch can reach the
    library: it must fall back instead of failing the step."""
    lib = object.__new__(BeamSharedPrefixFmha)
    calls = []
    lib._context_impl = SimpleNamespace(run_generation=calls.append)
    params = SimpleNamespace(
        attn=None, meta=SimpleNamespace(fmha_plan_caches={}), qkv_input=torch.zeros(1)
    )
    lib.run_generation(params)
    assert calls == [params]


@needs_flashinfer
def test_plan_step_rejects_an_invalid_prompt_page_index(monkeypatch):
    """Plan step rejects an invalid prompt page index."""
    monkeypatch.setenv("TLLM_FMHA_LIBS", f"+{LIB_NAME}")
    manager = _ManagerV2({7: [40, 41, -1, 43], 9: [50, 51, 52]})
    with pytest.raises(RuntimeError, match="invalid page index"):
        plan_step(_beam_meta(manager))


@needs_flashinfer
@pytest.mark.parametrize(
    "position_embedding_type, available",
    [
        (PositionEmbeddingType.learned_absolute, True),
        (PositionEmbeddingType.rope_gpt_neox, False),
        (PositionEmbeddingType.alibi, False),
    ],
)
def test_library_serves_only_models_that_apply_position_embeddings_themselves(
    position_embedding_type, available
):
    """The library implements neither RoPE nor ALiBi/relative bias, so any layer that needs one
    from the attention backend must be declined rather than run without it."""
    attn = SimpleNamespace(
        is_mla_enable=False,
        sparse_params=None,
        predicted_tokens_per_seq=1,
        attention_chunk_size=None,
        head_dim=64,
        num_heads=8,
        num_kv_heads=2,
        quant_mode=0,
        position_embedding_type=position_embedding_type,
    )
    assert BeamSharedPrefixFmha._is_available(attn) is available


def _qwen3_small_path():
    """The local Qwen3-0.6B checkpoint, or None when it is not available."""
    root = llm_models_root(check=False)
    if root is None:
        return None
    path = root / "Qwen3" / "Qwen3-0.6B"
    return path if path.exists() else None


def _run_beam_search(
    model_path, prompts, beam_width, max_tokens, env, max_num_tokens=None, iterations=None
):
    """Beam search over ``prompts``; ``iterations`` collects per-step batch kinds.

    ``max_num_tokens`` below the whole prompt set makes later requests prefill
    while earlier ones decode. ``iterations`` (needs the single-process worker)
    gets the number of generation steps that carried context requests (``mixed``)
    and that the library planned (``planned``); ``served`` counts the layer calls the library
    actually ran, so a test cannot pass by comparing the fallback with itself.
    """
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    plan_step = lib.plan_step
    run_generation = lib.BeamSharedPrefixFmha.run_generation
    if iterations is not None:
        iterations.update(mixed=0, planned=0, served=0)

        def counting_run_generation(self, params):
            """Count served generation calls."""
            iterations["served"] += 1
            return run_generation(self, params)

        lib.BeamSharedPrefixFmha.run_generation = counting_run_generation

        def counting_plan_step(metadata):
            """Count planned steps."""
            plan_step(metadata)
            iterations["mixed"] += int(metadata.num_contexts > 0 and metadata.num_generations > 0)
            state = metadata.fmha_plan_caches.get(LIB_NAME)
            iterations["planned"] += int(state is not None and state.plan is not None)

        lib.plan_step = counting_plan_step
    try:
        llm = LLM(
            model=str(model_path),
            max_beam_width=beam_width,
            max_batch_size=len(prompts),
            max_num_tokens=max_num_tokens or max(len(p) for p in prompts) * len(prompts),
            max_seq_len=max(len(p) for p in prompts) + max_tokens + 1,
            kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.3),
            cuda_graph_config=CudaGraphConfig(batch_sizes=[len(prompts)]),
            skip_tokenizer_init=True,
            # GEMM tactic choice changes bf16 numerics between processes.
            enable_autotuner=False,
        )
        with llm:
            sampling = SamplingParams(
                use_beam_search=True,
                n=beam_width,
                max_tokens=max_tokens,
                ignore_eos=True,
                end_id=-1,
                # Beam search returns only the sampled token's logprob.
                logprobs=0,
            )
            outputs = llm.generate(prompts, sampling_params=sampling)
        return [
            [(tuple(beam.token_ids), beam.cumulative_logprob) for beam in out.outputs]
            for out in outputs
        ]
    finally:
        lib.plan_step = plan_step
        lib.BeamSharedPrefixFmha.run_generation = run_generation
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _assert_beams_match(reference, candidate, beam_width, min_overlap=0.85):
    """Compare the candidate's beams with the default kernel's."""
    for ref_beams, cand_beams in zip(reference, candidate):
        assert len(cand_beams) == beam_width
        ref_tokens = [tokens for tokens, _ in ref_beams]
        cand_tokens = [tokens for tokens, _ in cand_beams]
        # Near-tied beams may swap in or out; the sets must still agree almost entirely.
        overlap = len(set(ref_tokens) & set(cand_tokens)) / beam_width
        assert overlap >= min_overlap, (overlap, ref_tokens, cand_tokens)
        assert cand_tokens[0] == ref_tokens[0]
        # Match beams by tokens, not rank.
        cand_logprobs = dict(cand_beams)
        for tokens, ref_lp in ref_beams:
            cand_lp = cand_logprobs.get(tokens)
            if ref_lp is not None and cand_lp is not None:
                assert abs(ref_lp - cand_lp) < 0.1 + 0.02 * abs(ref_lp)


_E2E_BASE_ENV = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}
_E2E_BEAM_WIDTH = 16
_E2E_PROMPT_LENGTHS = (300, 173)
_E2E_MAX_TOKENS = 3
# Prompts end 30 tokens into a 32-token KV block, so 5 generated tokens cross a block boundary.
_BOUNDARY_PROMPT_LENGTHS = (286, 190)
_BOUNDARY_MAX_TOKENS = 5


def _e2e_prompts(lengths):
    """Deterministic prompts of the given lengths."""
    vocab = 150000
    return [
        [1024 + ((17 * i + 13 * p) % (vocab - 1024)) for p in range(n)]
        for i, n in enumerate(lengths)
    ]


@functools.lru_cache(maxsize=2)
def _e2e_fallback_beams(model_path, lengths, max_tokens):
    """The fallback kernel's beams; the plain and FA4 tests share one run."""
    return _run_beam_search(
        model_path, _e2e_prompts(lengths), _E2E_BEAM_WIDTH, max_tokens, _E2E_BASE_ENV
    )


def _e2e_candidate(model_path, lengths, max_tokens, **extra_env):
    """Beam search with the library on; returns the beams and the library's call counts."""
    iterations = {}
    env = (
        _E2E_BASE_ENV
        | {
            "TLLM_FMHA_LIBS": f"+{LIB_NAME}",
            # Check mode validates the page table and prefix attention against dense references.
            "TLLM_BEAM_SHARED_PREFIX_CHECK": "1",
            "TLLM_BEAM_SHARED_PREFIX_MAX_TAIL": str(max_tokens - 1),
        }
        | extra_env
    )
    beams = _run_beam_search(
        model_path,
        _e2e_prompts(lengths),
        _E2E_BEAM_WIDTH,
        max_tokens,
        env,
        iterations=iterations,
    )
    assert iterations["served"] > 0, iterations
    return beams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(_qwen3_small_path() is None, reason="Qwen3-0.6B checkpoint not available")
@pytest.mark.threadleak(enabled=False)
def test_beam_shared_prefix_matches_fallback_e2e():
    """Beam search with the library reproduces the fallback kernel's beams.

    Three output tokens over a prompt that spans several KV blocks, two requests in one batch,
    CUDA graphs on: the second and third decode steps exercise the indirection-gathered tail, the
    first the empty tail.
    """
    model_path = _qwen3_small_path()
    candidate = _e2e_candidate(model_path, _E2E_PROMPT_LENGTHS, _E2E_MAX_TOKENS)
    reference = _e2e_fallback_beams(model_path, _E2E_PROMPT_LENGTHS, _E2E_MAX_TOKENS)
    _assert_beams_match(reference, candidate, _E2E_BEAM_WIDTH)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(_qwen3_small_path() is None, reason="Qwen3-0.6B checkpoint not available")
@pytest.mark.threadleak(enabled=False)
def test_beam_shared_prefix_tails_cross_a_kv_block_boundary_e2e():
    """Five generated tokens cross a KV block boundary, so later steps gather tails of up to four
    positions from two blocks. The fallback kernel's own beam sets differ between processes at this
    length (near-tied beams are pruned differently), so only gross disagreement fails here; the
    per-layer dense check (check mode) and the index unit test carry the exactness."""
    model_path = _qwen3_small_path()
    candidate = _e2e_candidate(model_path, _BOUNDARY_PROMPT_LENGTHS, _BOUNDARY_MAX_TOKENS)
    reference = _e2e_fallback_beams(model_path, _BOUNDARY_PROMPT_LENGTHS, _BOUNDARY_MAX_TOKENS)
    _assert_beams_match(reference, candidate, _E2E_BEAM_WIDTH, min_overlap=0.5)


def _fa4_prefix_available() -> bool:
    """Whether the FA4 prefix backend can run here."""
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._torch.attention.backends.fmha.beam_shared_prefix import _fa4_supported

    return _fa4_supported(128, torch.bfloat16, None)


@pytest.mark.skipif(not _fa4_prefix_available(), reason="needs SM90 and the FA4 kernels")
@pytest.mark.skipif(_qwen3_small_path() is None, reason="Qwen3-0.6B checkpoint not available")
@pytest.mark.threadleak(enabled=False)
def test_beam_shared_prefix_fa4_backend_matches_fallback_e2e():
    """The FlashAttention-4 prefix backend reproduces the fallback kernel's beams, graphs on."""
    model_path = _qwen3_small_path()
    candidate = _e2e_candidate(
        model_path,
        _E2E_PROMPT_LENGTHS,
        _E2E_MAX_TOKENS,
        TLLM_BEAM_SHARED_PREFIX_BACKEND="fa4",
    )
    # FA4 rounds differently, and beam search prunes at near-ties, so 1-2 of 16 beams may differ
    # (equally accurate against fp32). The top beam and shared beams' scores must still agree.
    reference = _e2e_fallback_beams(model_path, _E2E_PROMPT_LENGTHS, _E2E_MAX_TOKENS)
    _assert_beams_match(reference, candidate, _E2E_BEAM_WIDTH, min_overlap=0.75)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.skipif(_qwen3_small_path() is None, reason="Qwen3-0.6B checkpoint not available")
@pytest.mark.threadleak(enabled=False)
def test_beam_shared_prefix_matches_fallback_in_mixed_batches_e2e():
    """Generation rows that share a step with context requests use the library.

    The token budget fits only two of the four prompts per step, so the other two
    prefill while the first pair decodes. The context phase runs on the regular
    library and the generation rows on this one.
    """
    model_path = _qwen3_small_path()
    vocab = 150000
    prompts = [
        [1024 + ((17 * i + 13 * p) % (vocab - 1024)) for p in range(n)]
        for i, n in enumerate((300, 173, 260, 220))
    ]
    beam_width, max_tokens = 16, 3
    max_num_tokens = 300 + 173 + 2 * beam_width

    base_env = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1"}
    reference = _run_beam_search(
        model_path, prompts, beam_width, max_tokens, base_env, max_num_tokens=max_num_tokens
    )
    iterations = {}
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
        max_num_tokens=max_num_tokens,
        iterations=iterations,
    )

    assert iterations["served"] > 0, iterations
    assert iterations["mixed"] > 0, iterations
    assert iterations["planned"] >= iterations["mixed"], iterations
    _assert_beams_match(reference, candidate, beam_width)
