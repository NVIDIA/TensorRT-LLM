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
"""Tests for the opt-in sampling_shared_prefix FMHA library (n-sample requests with a linear draft,
KV cache manager V2, SM90).

The grouping and the cost model run on the host; the planner and the FlashAttention-4 kernels need
an SM90 GPU and are checked against float64 references; the end-to-end tests run EAGLE-3 n-sampling
with and without the library and need the checkpoints named by ``TLLM_TEST_EAGLE3_TARGET`` /
``TLLM_TEST_EAGLE3_DRAFT``.
"""

import contextlib
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.attention.backends.fmha import sampling_shared_prefix as lib


def _fa4_available():
    if not torch.cuda.is_available():
        return False
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fa4 import fa4_supported

    return fa4_supported(128, torch.bfloat16)


needs_fa4 = pytest.mark.skipif(not _fa4_available(), reason="needs SM90 and FlashAttention-4")


@pytest.fixture(autouse=True)
def _open_size_gate(monkeypatch):
    """The tests use short prompts; keep every group of two or more rows unless a test says otherwise."""
    monkeypatch.setenv(lib.MIN_SHARED_ENV, "0")
    monkeypatch.setenv(lib.MIN_GROUP_ENV, "2")


@pytest.fixture(autouse=True)
def _model_shape(request, monkeypatch):
    """The library remembers the attention shape of the model it serves; the host and planner tests
    use the MHA shape of the n-sample target (20 heads of 128). The end-to-end tests start without
    one, so the model they load sets its own."""
    shape = None if request.node.name.endswith("_e2e") else (20, 20, 128)
    monkeypatch.setattr(lib, "_model_shape", shape)


def _build(pages, kv_lens, tpb, *, q_len=1, max_sharing_groups=None, padded_groups=None, **kw):
    pages = np.asarray(pages, dtype=np.int64)
    return lib._build_grouping(
        1,
        np.arange(pages.shape[0]),
        pages,
        np.asarray(kv_lens),
        tpb,
        max_sharing_groups=max_sharing_groups,
        padded_groups=padded_groups,
        q_len=q_len,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Grouping (host)
# --------------------------------------------------------------------------- #
def test_grouping_groups_rows_with_common_leading_pages():
    tpb = 4
    # Rows 0, 2, 3 share pages 5, 6 (a third page differs); row 1 is alone;
    # rows 2, 3 also share page 7 but row 2's first token of the step sits in it.
    tables = [[5, 6, 10, 0], [20, 21, 22, 0], [5, 6, 7, 0], [5, 6, 7, 11]]
    grouping = _build(tables, [10, 9, 12, 14], tpb)
    # Groups in order of first appearance: {0, 2, 3} then the placeholder {1}.
    assert grouping.perm.tolist() == [0, 2, 3, 1]
    assert not grouping.is_identity
    assert grouping.qo_indptr.tolist() == [0, 3, 4]
    assert grouping.prefix_indptr.tolist() == [0, 2, 3]
    assert grouping.prefix_indices.tolist()[:2] == [5, 6]
    assert grouping.prefix_last_page_len.tolist() == [tpb, 1]
    assert grouping.prefix_empty.tolist() == [False, False, False, True]
    assert grouping.shared_g.tolist() == [2, 2, 2, 0]
    assert grouping.shared_tokens == 3 * 2 * tpb


def test_grouping_with_draft_tokens_shares_only_pages_before_the_step():
    """With ``q_len`` tokens per row, the shared run ends before the page of the row's first step token."""
    tpb, q_len = 4, 4
    tables = [[5, 6, 7, 30], [5, 6, 7, 31]]
    # kv 14: the step's tokens are positions 10-13, the first in page 2.
    grouping = _build(tables, [14, 14], tpb, q_len=q_len)
    assert grouping.prefix_indptr.tolist() == [0, 2]
    assert grouping.qo_indptr.tolist() == [0, 2 * q_len]
    assert grouping.private_from.tolist() == [2, 2] and grouping.last_pages.tolist() == [3, 3]


def test_grouping_pads_groups_and_demotes_surplus_sharing_groups():
    """A batch size gets a fixed number of prefix groups: the sharing groups that save the most page
    reads, the placeholder group (also without rows) and empty padding groups."""
    tpb = 4
    # Group {0, 1} shares two pages, group {2, 3, 4} shares one page (the
    # current token of row 4 sits in the second), row 5 shares nothing.
    tables = [[1, 2, 9], [1, 2, 10], [3, 4, 11], [3, 4, 12], [3, 13, 0], [5, 14, 0]]
    kv_lens = [10, 11, 9, 12, 7, 6]
    grouping = _build(tables, kv_lens, tpb, padded_groups=4)
    assert grouping.num_sharing == 2 and grouping.num_groups == 4
    assert grouping.perm.tolist() == [0, 1, 2, 3, 4, 5]
    assert grouping.qo_indptr.tolist() == [0, 2, 5, 6, 6]
    assert grouping.prefix_indptr.tolist() == [0, 2, 3, 4, 5]
    assert grouping.prefix_indices.tolist()[:3] == [1, 2, 3]
    assert grouping.prefix_last_page_len.tolist() == [tpb, tpb, 1, 1]
    assert grouping.prefix_empty.tolist() == [False] * 5 + [True]

    # Room for one sharing group: {0, 1} and {2, 3, 4} both save two page
    # reads; the tie goes to the group with the lower first page.
    grouping = _build(tables, kv_lens, tpb, max_sharing_groups=1, padded_groups=2)
    assert grouping.num_sharing == 1
    assert grouping.qo_indptr.tolist() == [0, 2, 6]
    assert grouping.prefix_empty.tolist() == [False, False] + [True] * 4
    assert grouping.shared_g.tolist() == [2, 2, 0, 0, 0, 0]

    # No sharing at all still yields the placeholder group plus padding.
    grouping = _build([tables[i] for i in (0, 2, 5)], [10, 9, 6], tpb, padded_groups=3)
    assert grouping.num_sharing == 0
    assert grouping.qo_indptr.tolist() == [0, 3, 3, 3]
    assert grouping.prefix_empty.all()


def test_grouping_drops_groups_that_do_not_pay():
    """``group_pays`` sees each candidate group's rows and shared tokens; the rows of the
    groups it rejects take the tail alone."""
    tpb = 4
    tables = [[1, 2, 9], [1, 2, 10], [3, 4, 11], [3, 4, 12], [3, 4, 13]]
    seen = []

    def pays(rows, tokens):
        seen.append((rows.tolist(), tokens.tolist()))
        return rows >= 3

    grouping = _build(tables, [10, 11, 10, 11, 12], tpb, padded_groups=3, group_pays=pays)
    assert seen == [([2, 3], [2 * tpb, 2 * tpb])]
    assert grouping.num_sharing == 1
    assert grouping.perm.tolist() == [2, 3, 4, 0, 1]
    assert grouping.prefix_empty.tolist() == [False] * 3 + [True] * 2
    assert grouping.shared_g.tolist() == [2, 2, 2, 0, 0]


def test_grouping_advances_across_page_boundaries_unless_a_capped_run_may_grow():
    """Rows crossing a page boundary keep their groups, unless a group's shared run was limited
    by its members' full pages."""
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
    grouping = _build(tables, [15, 16, 14, 13, 6, 9], tpb, padded_groups=4)
    assert grouping.num_sharing == 2 and grouping.capped_keys.tolist() == [1]

    # Rows 0 and 5 cross a boundary; group {3, 4} keeps its cap.
    after = np.array([17, 16, 14, 13, 6, 10])
    advanced = lib._advance(grouping, (after - 1) // tpb, (after - 1) // tpb)
    rebuilt = _build(tables, after, tpb, padded_groups=4)
    assert advanced is not None and advanced.version == grouping.version
    for name in (
        "perm",
        "prefix_empty",
        "shared_g",
        "qo_indptr",
        "prefix_indptr",
        "prefix_indices",
    ):
        assert getattr(advanced, name).tolist() == getattr(rebuilt, name).tolist(), name
    assert advanced.private_from.tolist() == rebuilt.private_from.tolist()

    # Row 4's current token moves to its third page: group {3, 4} may now share
    # its second page, so the groups have to be rebuilt.
    after = np.array([15, 16, 14, 13, 9, 9])
    assert lib._advance(grouping, (after - 1) // tpb, (after - 1) // tpb) is None


# --------------------------------------------------------------------------- #
# Cost model (host)
# --------------------------------------------------------------------------- #
def test_size_gate_follows_saved_bytes(monkeypatch):
    """A group shares when the KV bytes it saves per layer reach the threshold of its attention
    type: MHA needs wide groups or long prefixes, GQA shares every group (see ``COST_MODEL``)."""
    monkeypatch.delenv(lib.MIN_SHARED_ENV)
    mha, gqa = (20, 20, 128), (28, 4, 128)
    rows, tokens = np.array([2, 2, 4, 8, 40, 20]), np.array([32, 1600, 128, 320, 4000, 600])
    # MHA: 10,240 bytes per token; 8 MB is 781 token-rows not read again.
    assert lib.group_saved_bytes(rows, tokens, mha).tolist() == [
        (r - 1) * t * 10240 for r, t in zip(rows.tolist(), tokens.tolist())
    ]
    monkeypatch.setattr(lib, "_model_shape", mha)
    assert lib._group_filter(9)(rows, tokens).tolist() == [False, True, False, True, True, True]
    monkeypatch.setattr(lib, "_model_shape", gqa)
    assert lib._group_filter(9)(rows, tokens).all()
    # The Curiosity-4B cells (20-80 rows per group, 600-6,500 shared tokens) always share.
    monkeypatch.setattr(lib, "_model_shape", mha)
    cells = np.array([[20, 600], [40, 600], [80, 600], [20, 6500]])
    assert lib._group_filter(9)(cells[:, 0], cells[:, 1]).all()


def test_group_filter_follows_the_env_threshold(monkeypatch):
    rows, tokens = np.array([2, 40, 40, 1]), np.array([4000, 32, 4000, 4000])
    monkeypatch.setenv(lib.MIN_SHARED_ENV, "1000")
    assert lib._group_filter(9)(rows, tokens).tolist() == [True, False, True, False]
    monkeypatch.setenv(lib.MIN_GROUP_ENV, "3")
    assert lib._group_filter(9)(rows, tokens).tolist() == [False, False, True, False]


# --------------------------------------------------------------------------- #
# Batch eligibility (host)
# --------------------------------------------------------------------------- #
def _eligible_meta(manager, beam_width=1):
    return SimpleNamespace(
        beam_width=beam_width,
        num_generations=8,
        num_contexts=0,
        is_cross=False,
        enable_helix=False,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        kv_cache_block_offsets=torch.empty(0),
        kv_cache_params=SimpleNamespace(use_cache=True, num_extra_kv_tokens=0),
        kv_cache_manager=manager,
        fmha_plan_caches={},
        is_cuda_graph=False,
    )


def test_beam_search_batches_and_a_v1_manager_are_declined():
    """n-sample batches on the V2 manager only: with ``beam_width > 1`` or a V1 manager (no index
    mapper) the library plans nothing and the regular kernel runs."""
    manager = SimpleNamespace(
        get_buffers=lambda layer, kv_layout: None,
        host_kv_cache_block_offsets=None,
        kv_factor=2,
        tokens_per_block=32,
    )
    meta = _eligible_meta(manager)
    assert not lib.sampling_metadata_eligible(meta)  # V1: no index mapper
    lib.plan_step(meta)
    assert meta.fmha_plan_caches == {} and not lib.has_active_plan(meta)
    manager.index_mapper = object()
    assert lib.sampling_metadata_eligible(meta)
    beam = _eligible_meta(manager, beam_width=4)
    assert not lib.sampling_metadata_eligible(beam)
    lib.plan_step(beam)
    assert beam.fmha_plan_caches == {} and not lib.has_active_plan(beam)


def test_per_layer_page_tables_are_declined():
    """With per-layer page tables (SWA scratch reuse) the block-offset table is indexed by local
    layer, and two layers of one lifecycle group can map different pages; the library reads the
    group's table for every layer, so it declines such a manager."""
    manager = SimpleNamespace(
        get_buffers=lambda layer, kv_layout: None,
        host_kv_cache_block_offsets=None,
        kv_factor=2,
        tokens_per_block=32,
        index_mapper=object(),
        _use_per_layer_page_tables=False,
    )
    assert lib.sampling_metadata_eligible(_eligible_meta(manager))
    manager._use_per_layer_page_tables = True
    meta = _eligible_meta(manager)
    assert not lib.sampling_metadata_eligible(meta)
    lib.plan_step(meta)
    assert meta.fmha_plan_caches == {} and not lib.has_active_plan(meta)


# --------------------------------------------------------------------------- #
# Planner (SM90)
# --------------------------------------------------------------------------- #
class _Manager(SimpleNamespace):  # weak-referenceable, like the real managers
    pass


def _planner_meta(host, request_ids, slot_of, kv_lens, tpb, *, q_len, caches, index_scale=2,
                  kv_factor=2, is_cuda_graph=False):  # fmt: skip
    rows = len(request_ids)
    manager = _Manager(
        layer_offsets={0: 0},
        kv_factor=kv_factor,
        max_beam_width=1,
        index_mapper=SimpleNamespace(get_index=lambda rid: slot_of[rid]),
        layer_to_pool_mapping_dict={0: 0},
        get_layer_page_index_scale=lambda layer_idx: index_scale,
        get_buffers=lambda layer, kv_layout: torch.empty(512, 1),
    )
    return SimpleNamespace(
        fmha_plan_caches=caches,
        kv_cache_manager=manager,
        request_ids=request_ids,
        host_kv_cache_block_offsets=host,
        kv_cache_block_offsets=torch.empty((0,), device="cuda"),
        kv_lens_runtime=torch.tensor(kv_lens),
        tokens_per_block=tpb,
        num_generations=rows,
        num_tokens=rows * q_len,
        seq_lens=torch.full((rows,), q_len, dtype=torch.int32),
        is_cuda_graph=is_cuda_graph,
    )


@needs_fa4
def test_planner_reads_kv_manager_v2_block_table():
    """V2 keeps base page indices by request slot; the planner rescales them to the pages of the
    layer's ``get_buffers`` view and uploads the grouping to the step buffers."""
    tpb, q_len, index_scale, kv_factor = 4, 4, 4, 2
    # Base page indices per request slot; slot order differs from batch order.
    slot_tables = {3: [5, 6, 10, 11], 0: [5, 6, 7, 12], 1: [20, 21, 22, 23]}
    request_ids = [101, 102, 103]  # batch order -> slots 3, 0, 1
    slot_of = {101: 3, 102: 0, 103: 1}
    host = torch.zeros(1, 4, 2, 4, dtype=torch.int32)
    for slot, table in slot_tables.items():
        host[0, slot, 0, :] = torch.tensor(table)
    caches = {}
    meta = _planner_meta(host, request_ids, slot_of, [12, 14, 13], tpb, q_len=q_len, caches=caches,
                         index_scale=index_scale, kv_factor=kv_factor)  # fmt: skip
    lib._plan_sampling_step(meta)
    state = caches[lib.SAMPLING_KEY]
    assert state.plan is not None
    grouping = state.plan.grouping
    pages_per_base = index_scale // kv_factor
    # Rows 0 and 1 share two base pages; row 2 stands alone.
    assert grouping.is_identity
    assert grouping.qo_indptr.tolist()[:3] == [0, 2 * q_len, 3 * q_len]
    assert grouping.prefix_indices.tolist()[:2] == [5 * pages_per_base, 6 * pages_per_base]
    torch.cuda.synchronize()
    buffers = state.buffers
    assert buffers.perm.tolist() == [0, 1, 2]
    assert buffers.shared.tolist() == [2, 2, 0]
    assert buffers.empty.tolist() == [False, False, True]


@needs_fa4
@pytest.mark.parametrize("new_request_ids", [True, False], ids=["new_requests", "same_requests"])
def test_planner_follows_new_pages_in_recycled_slots(new_request_ids):
    """Rows whose slots, first pages and lengths stay the same but whose shared pages change get a new
    grouping: new requests in recycled slots (``new_requests``), or the same requests with their pages
    released and allocated again (``same_requests``, e.g. preemption). The plan must match one built
    from scratch for the new tables."""
    tpb, q_len = 4, 4
    host = torch.zeros(1, 2, 2, 4, dtype=torch.int32)

    def meta(request_ids, slot_of, caches):
        # 12 tokens: the step's tokens are positions 8-11 (page 2); pages 0-1 may be shared.
        return _planner_meta(host, request_ids, slot_of, [12, 12], tpb, q_len=q_len,
                             caches=caches, index_scale=1, kv_factor=1)  # fmt: skip

    host[0, 0, 0, :3] = torch.tensor([5, 6, 7])
    host[0, 1, 0, :3] = torch.tensor([5, 6, 8])
    caches = {}
    lib._plan_sampling_step(meta([101, 102], {101: 0, 102: 1}, caches))
    assert caches[lib.SAMPLING_KEY].plan.grouping.prefix_indices.tolist()[:2] == [5, 6]

    # The slots are reused; the first prompt page hits the reuse cache, the second is new.
    host[0, 0, 0, :3] = torch.tensor([5, 9, 30])
    host[0, 1, 0, :3] = torch.tensor([5, 9, 31])
    request_ids = [201, 202] if new_request_ids else [101, 102]
    slot_of = dict(zip(request_ids, (0, 1)))
    lib._plan_sampling_step(meta(request_ids, slot_of, caches))
    state = caches[lib.SAMPLING_KEY]
    fresh_caches = {}
    lib._plan_sampling_step(meta(request_ids, slot_of, fresh_caches))
    fresh = fresh_caches[lib.SAMPLING_KEY].plan.grouping
    assert state.plan.grouping.prefix_indices.tolist()[:2] == [5, 9]
    assert state.plan.grouping.prefix_indices.tolist() == fresh.prefix_indices.tolist()

    # Entries past a row's last page were never read: changing them keeps the grouping.
    version = state.grouping.version
    host[0, 0, 0, 3] = 40
    lib._plan_sampling_step(meta(request_ids, slot_of, caches))
    assert state.grouping.version == version


@needs_fa4
def test_planner_declines_short_drafts_and_eager_batches_without_sharing(monkeypatch):
    tpb = 4
    host = torch.zeros(1, 3, 2, 4, dtype=torch.int32)
    host[0, 0, 0, :] = torch.tensor([1, 2, 3, 4])
    host[0, 1, 0, :] = torch.tensor([5, 6, 7, 8])
    host[0, 2, 0, :] = torch.tensor([9, 10, 11, 12])
    ids, slots = [0, 1, 2], {0: 0, 1: 1, 2: 2}
    # Fewer query tokens per row than the FA4 prefix tile is measured with.
    caches = {}
    lib._plan_sampling_step(_planner_meta(host, ids, slots, [12] * 3, tpb, q_len=lib.MIN_Q_LEN - 1,
                                          caches=caches))  # fmt: skip
    assert lib.SAMPLING_KEY not in caches
    # No two rows share a page: an eager batch is declined, a CUDA graph batch is planned.
    caches = {}
    lib._plan_sampling_step(_planner_meta(host, ids, slots, [12] * 3, tpb, q_len=4, caches=caches))
    assert caches[lib.SAMPLING_KEY].plan is None
    caches = {}
    lib._plan_sampling_step(_planner_meta(host, ids, slots, [12] * 3, tpb, q_len=4, caches=caches,
                                          is_cuda_graph=True))  # fmt: skip
    plan = caches[lib.SAMPLING_KEY].plan
    assert plan is not None and plan.grouping.num_sharing == 0
    assert plan.grouping.prefix_empty.all()


def _step_tables(prompts, children, private_pages, first_private, gen_lens, tpb):
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


class _Attn(SimpleNamespace):  # weak-referenceable, like the real attention layers
    def out_head_size(self, is_gen_only):
        return self.head_dim


@needs_fa4
def test_library_cuda_graph_replay_matches_eager_across_steps(monkeypatch):
    """Capture one CUDA graph around the library's generation call of a layer (``run_generation``:
    per-step tail tables, K/V write, FA4 prefix, tail and merge) and replay it over steps whose
    plans differ: tails that grow inside a page and across a page boundary, fewer and smaller
    groups, surplus sharing groups, a batch without any sharing, and padded dummy rows.

    Each replay must equal an eager call on the same planned buffers bit for bit (the graph reads
    the refreshed tables, not the captured ones), write the step's K/V like it, and match a dense
    float32 reference over each row's whole page table."""
    from tensorrt_llm.functional import PositionEmbeddingType

    monkeypatch.setenv(lib.MAX_GROUPS_ENV, "3")
    device = torch.device("cuda")
    num_heads, num_kv_heads, head_dim, q_len = 20, 20, 128, 4
    tpb, rows, num_pages, max_blocks = 16, 12, 160, 12
    sm_scale = 1.0 / math.sqrt(head_dim)
    gen = torch.Generator(device=device).manual_seed(11)
    pool = torch.randn(num_pages, 2, num_kv_heads, tpb, head_dim, generator=gen, device=device)
    pool = pool.bfloat16()
    eager_pool = pool.clone()

    # Prompt pages 0-29; private pages from 40 on, 8 per row.
    steps = []
    # 1. Three prompts x four children with ragged prompts and tails.
    steps.append(_step_tables([(0, 70), (10, 37), (20, 100)], [4, 4, 4], 8, 40, [5, 7], tpb))
    # 2. Same rows, every tail one token longer (same pages).
    steps.append((steps[0][0], [n + 1 for n in steps[0][1]]))
    # 3. Tails across a page boundary.
    steps.append(_step_tables([(0, 70), (10, 37), (20, 100)], [4, 4, 4], 8, 40, [16, 18], tpb))
    # 4. Fewer, smaller groups: 2 x 3 children, the rest unrelated prompts.
    tables, kv_lens = _step_tables([(0, 70), (20, 100)], [3, 3], 8, 40, [9], tpb)
    for r in range(6):
        tables.append([30 + r] + list(range(100 + 8 * r, 104 + 8 * r)))
        kv_lens.append(44 + r)
    steps.append((tables, kv_lens))
    # 5. Four sharing groups for three group slots: the smallest saving is demoted (a tie goes
    # to the lower first page, so the group of the prompt at page 26 joins the placeholder).
    steps.append(
        _step_tables([(0, 70), (10, 37), (20, 100), (26, 40)], [3, 3, 3, 3], 8, 40, [6], tpb)
    )
    # 6. No sharing at all.
    steps.append(
        ([[30 + r, 100 + 4 * r, 101 + 4 * r] for r in range(rows)], [37 + r for r in range(rows)])
    )
    # 7. Two real groups and padded dummy rows that all map one page.
    tables, kv_lens = _step_tables([(0, 70), (10, 37)], [4, 4], 8, 40, [6], tpb)
    for _ in range(4):
        tables.append([150])
        kv_lens.append(q_len)
    steps.append((tables, kv_lens))
    # (sharing groups, rows without a shared prefix) of every step.
    expected_groups = [(3, 0), (3, 0), (3, 0), (2, 6), (3, 3), (0, 12), (2, 4)]

    # Fixed-address step inputs, refreshed per step like the metadata of a captured batch size.
    offsets = torch.zeros(1, rows, 2, max_blocks, dtype=torch.int32, device=device)
    kv_lens_device = torch.zeros(rows, dtype=torch.int32, device=device)
    width = (num_heads + 2 * num_kv_heads) * head_dim
    qkv_static = torch.zeros(rows * q_len, width, dtype=torch.bfloat16, device=device)
    out_static = torch.zeros(
        rows * q_len, num_heads * head_dim, dtype=torch.bfloat16, device=device
    )
    caches = {}
    attn = _Attn(
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, layer_idx=0,
        q_scaling=1.0, is_mla_enable=False,
        position_embedding_type=int(PositionEmbeddingType.learned_absolute), rotary_cos_sin=None,
    )  # fmt: skip
    # One manager for every step, as in a real run; ``pool_of`` picks the pool its view returns.
    pool_of = {"pool": pool}
    manager = _Manager(
        layer_offsets={0: 0},
        kv_factor=2,
        max_beam_width=1,
        index_mapper=SimpleNamespace(get_index=lambda rid: rid),
        layer_to_pool_mapping_dict={0: 0},
        get_layer_page_index_scale=lambda layer_idx: 2,
        get_buffers=lambda layer, kv_layout: pool_of["pool"],
    )
    graph_lib = lib.SamplingSharedPrefixFmha(attn)
    eager_lib = lib.SamplingSharedPrefixFmha(attn)
    graph = None
    for step, (tables, kv_lens) in enumerate(steps):
        assert len(tables) == rows
        host = torch.zeros(1, rows, 2, max_blocks, dtype=torch.int32)
        for r, table in enumerate(tables):
            host[0, r, 0, : len(table)] = torch.tensor(table)
        offsets.copy_(host * 2)  # device entries: pool-view page times the KV factor
        kv_lens_device.copy_(torch.tensor(kv_lens, dtype=torch.int32))
        meta = SimpleNamespace(
            fmha_plan_caches=caches,
            kv_cache_manager=manager,
            request_ids=list(range(rows)),
            host_kv_cache_block_offsets=host,
            kv_cache_block_offsets=offsets,
            kv_lens_runtime=torch.tensor(kv_lens),
            kv_lens_cuda_runtime=kv_lens_device,
            tokens_per_block=tpb,
            num_generations=rows,
            num_tokens=rows * q_len,
            seq_lens=torch.full((rows,), q_len, dtype=torch.int32),
            is_cuda_graph=True,
        )
        lib._plan_sampling_step(meta)
        state = caches[lib.SAMPLING_KEY]
        assert state.plan is not None
        grouping = state.plan.grouping
        assert (grouping.num_sharing, grouping.num_placeholder_rows) == expected_groups[step]
        assert grouping.num_groups == state.buffers.num_groups == 4

        qkv = torch.randn(rows * q_len, width, generator=gen, device=device).bfloat16()
        dummy_rows = [r for r in range(rows) if tables[r] == [150]]
        for r in dummy_rows:
            # Padding rows repeat one request: identical inputs, the same KV slots.
            qkv[r * q_len : (r + 1) * q_len] = qkv[
                dummy_rows[0] * q_len : (dummy_rows[0] + 1) * q_len
            ]

        def params(qkv_in, out):
            return SimpleNamespace(qkv_input=qkv_in, attn=attn, meta=meta, num_tokens=rows * q_len,
                                   output=out)  # fmt: skip

        # The eager call works on a copy of the pool (its layer view is fetched again).
        eager_pool.copy_(pool)
        eager_out = torch.zeros_like(out_static)
        pool_of["pool"] = eager_pool
        eager_lib._pool_views.clear()
        eager_lib.run_generation(params(qkv.clone(), eager_out))
        pool_of["pool"] = pool
        qkv_static.copy_(qkv)
        if graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_lib.run_generation(params(qkv_static, out_static))
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(out_static, eager_out), f"step {step}: replay differs from eager"
        assert torch.equal(pool, eager_pool), f"step {step}: replay wrote different K/V"
        q = qkv[:, : num_heads * head_dim].view(rows * q_len, num_heads, head_dim)
        out = out_static.view(rows * q_len, num_heads, head_dim)
        for r in range(rows):
            if r in dummy_rows:
                continue
            pages = torch.tensor(tables[r], device=device)
            kv = pool[pages].permute(0, 3, 1, 2, 4).reshape(-1, 2, num_kv_heads, head_dim)
            for j in range(q_len):
                visible = kv[: kv_lens[r] - q_len + j + 1].float()
                qr = q[r * q_len + j].float().view(num_kv_heads, -1, head_dim)
                scores = torch.einsum("kgd,tkd->kgt", qr, visible[:, 0]) * sm_scale
                expected = torch.einsum(
                    "kgt,tkd->kgd", torch.softmax(scores, dim=-1), visible[:, 1]
                ).reshape(num_heads, head_dim)
                torch.testing.assert_close(out[r * q_len + j].float(), expected, atol=2e-2,
                                           rtol=2e-2, msg=f"step {step} row {r} token {j}")  # fmt: skip


# --------------------------------------------------------------------------- #
# FlashAttention-4 kernels (SM90)
# --------------------------------------------------------------------------- #
def _spec_case(num_heads, num_kv_heads, head_dim, q_len, shuffled, placeholder_rows, seed,
               overshoot, massive=False):  # fmt: skip
    """Tables, tensors and float64 reference of one speculative step: 3 groups of 5 rows sharing 4
    pages each, plus ``placeholder_rows`` rows without a shared prefix.

    ``cached[r]`` private tokens precede the step's ``q_len`` tokens. With ``overshoot`` the host page
    table of every row is one page longer than the device length needs, as under the overlap
    scheduler.
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
    dgen = torch.Generator(device=device).manual_seed(seed)
    pool = (
        0.7 * torch.randn(num_pages, 2, num_kv_heads, tpb, head_dim, generator=dgen, device=device)
    ).bfloat16()
    ntok = rows * q_len
    q = torch.randn(ntok, num_heads, head_dim, generator=dgen, device=device).bfloat16()
    k = torch.randn(ntok, num_kv_heads, head_dim, generator=dgen, device=device).bfloat16()
    v = torch.randn(ntok, num_kv_heads, head_dim, generator=dgen, device=device).bfloat16()
    perm = torch.randperm(rows, generator=gen) if shuffled else torch.arange(rows)
    placeholder_page = groups * prefix_pages
    page, own = placeholder_page + 1, []
    for b in blocks:
        own.append(list(range(page, page + b)))
        page += b
    if massive:
        # Massive-activation magnitudes of real models (Qwen2.5-7B layer 27: |k| ~ 420, |q| ~ 66, scaled logits
        # of -3500..-6800 for some heads): keys of one sign, queries of either sign per head, so odd heads have
        # every logit hugely negative and even heads random-signed.
        pool[:, 0] = (pool[:, 0].float().abs() * 600.0).bfloat16()
        sign = torch.where(torch.arange(num_heads, device=device) % 2 == 1, -1.0, 1.0)[
            None, :, None
        ]
        q = (q.float().abs() * 80.0 * sign).bfloat16()
        q[:, ::2] = (torch.randn(q[:, ::2].shape, generator=dgen, device=device) * 80.0).bfloat16()
        k = (k.float().abs() * 600.0).bfloat16()
    qo = [g * per_group * q_len for g in range(groups + 1)] + [
        (rows if placeholder_rows else groups * per_group) * q_len
    ]
    grouping = SimpleNamespace(
        num_groups=groups + 1,
        qo_indptr=np.array(qo, dtype=np.int32),
        prefix_indptr=np.array(
            [g * prefix_pages for g in range(groups + 1)] + [groups * prefix_pages + 1], np.int32
        ),
        prefix_indices=np.array(list(range(groups * prefix_pages)) + [placeholder_page], np.int32),
        prefix_last_page_len=np.array([tpb] * groups + [1], np.int32),
    )
    empty = [False] * sharing_rows + [True] * placeholder_rows
    # Dense reference in float64 on the pre-step pool.
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
            probs = torch.softmax(torch.einsum("hgd,htd->hgt", qh, keys) / math.sqrt(head_dim), -1)
            ref[b * q_len + t] = torch.einsum("hgt,htd->hgd", probs, vals).reshape(
                num_heads, head_dim
            )
    return dict(
        pool=pool, q=q, k=k, v=v, ref=ref, grouping=grouping, rows=rows, cached=cached, own=own,
        tpb=tpb, per_group=per_group, prefix_pages=prefix_pages, ntok=ntok,
        sharing_rows=sharing_rows, perm=perm.to(device, torch.int64),
        empty=torch.tensor(empty, device=device),
    )  # fmt: skip


def _make_prefix(case, num_heads, head_dim, q_len, splits=1):
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fa4 import Fa4Prefix

    groups = case["grouping"].num_groups
    requests = Fa4Prefix.num_requests_for(groups + 1, case["ntok"], q_len) * splits
    max_pages = 8
    prefix = Fa4Prefix(case["ntok"], num_heads, head_dim, q_len, requests, max_pages,
                       torch.bfloat16, torch.device("cuda"), case["tpb"], splits=splits)  # fmt: skip
    tables = lib._prefix_tables(
        case["grouping"], case["tpb"], prefix.chunk_tokens, requests, max_pages, splits=splits
    )
    for dst, host in zip(prefix.table_arrays(), tables):
        dst.copy_(torch.from_numpy(host))
    return prefix


def _make_tail(case, num_heads, head_dim, q_len, kv_factor=2, splits=1):
    """``Fa4Tail`` of a case with its step tables built from a device block-offset table (batch
    order, ``kv_factor`` entries per pool-view page, a row's shared pages first)."""
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fa4 import Fa4Tail

    device = torch.device("cuda")
    rows, tpb, prefix_pages = case["rows"], case["tpb"], case["prefix_pages"]
    perm = case["perm"].tolist()
    max_blocks = 64
    offsets = torch.full((rows, max_blocks), -7, dtype=torch.int32)
    shared, kv_lens = [], [0] * rows
    for i in range(rows):
        b = perm[i]
        sharing = i < case["sharing_rows"]
        g = i // case["per_group"]
        pages = (list(range(g * prefix_pages, (g + 1) * prefix_pages)) if sharing else []) + case[
            "own"
        ][i]
        offsets[b, : len(pages)] = torch.tensor(pages, dtype=torch.int32) * kv_factor
        shared.append(prefix_pages if sharing else 0)
        kv_lens[b] = shared[-1] * tpb + case["cached"][i] + q_len
    tail = Fa4Tail(
        rows, q_len, num_heads, head_dim, max_blocks, torch.bfloat16, device, splits=splits
    )
    tail.build_tables(
        offsets.to(device),
        case["perm"],
        torch.tensor(shared, dtype=torch.int32, device=device),
        torch.tensor(kv_lens, dtype=torch.int32, device=device),
        kv_factor,
        tpb,
    )
    torch.cuda.synchronize()
    for i in range(rows):
        b = perm[i]
        own = case["own"][i]
        assert tail.table[b, : len(own)].tolist() == own
        assert int(tail.tail_lens[b]) == case["cached"][i] + q_len
    return tail


def _run(case, pool, q, k, v, out, prefix, tail, head_dim):
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fa4 import (
        fa4_sampling_attention,
    )

    fa4_sampling_attention(pool, q, k, v, out, perm=case["perm"], prefix_empty=case["empty"],
                           prefix=prefix, tail=tail, sm_scale=1.0 / math.sqrt(head_dim))  # fmt: skip


@needs_fa4
@pytest.mark.parametrize(
    "num_heads,num_kv_heads,head_dim,q_len,shuffled,placeholder_rows,overshoot",
    [
        (28, 4, 128, 4, True, 2, True),
        (28, 4, 128, 9, True, 2, True),
        (20, 20, 128, 4, False, 0, False),
        (20, 20, 128, 9, True, 2, False),
    ],
)
@pytest.mark.parametrize("massive", [False, True])
@pytest.mark.parametrize(
    "prefix_splits,tail_splits,min_split_tokens",
    [(1, 1, 512), (3, 1, 32), (8, 1, 32), (1, 4, 32), (8, 16, 32), (8, 16, 512)],
)
def test_fa4_sampling_attention_matches_dense_and_writes_kv(
    num_heads, num_kv_heads, head_dim, q_len, shuffled, placeholder_rows, overshoot, massive,
    prefix_splits, tail_splits, min_split_tokens, monkeypatch,
):  # fmt: skip
    """FlashAttention-4 for the shared pages (whole, or split into runs, some empty: 4 shared pages
    split 8 ways) and the private pages + their merge against the float64 reference (also with
    massive activations), and the step's K/V written at cached + j."""
    from tensorrt_llm._torch.attention.backends.fmha import flash_attn4_generation as fa4_gen

    # One page (32 tokens) per run/chunk at least: the 4 shared pages really split.
    monkeypatch.setattr(fa4_gen, "MIN_SPLIT_TOKENS", min_split_tokens)
    for seed in range(2):
        case = _spec_case(num_heads, num_kv_heads, head_dim, q_len, shuffled, placeholder_rows,
                          seed, overshoot, massive=massive)  # fmt: skip
        prefix = _make_prefix(case, num_heads, head_dim, q_len, splits=prefix_splits)
        tail = _make_tail(case, num_heads, head_dim, q_len, splits=tail_splits)
        pool = case["pool"]
        before = pool.clone()
        out = torch.empty_like(case["q"])
        _run(case, pool, case["q"], case["k"], case["v"], out, prefix, tail, head_dim)
        torch.cuda.synchronize()
        ref = case["ref"]
        if massive:
            head_max = ref.abs().amax(-1)
            err = (out.double() - ref).abs().amax(-1)
            assert not bool(((out.abs().amax(-1) == 0) & (head_max > 1e-3)).any()), "zero head"
            assert float((err / head_max.clamp(min=1e-3)).max()) < 2.5e-2
        else:
            err = (out.double() - ref).abs().amax() / ref.abs().amax()
            assert float(err) < 2e-2, f"seed {seed}: rel err {float(err)}"
        expect = before.clone()
        for i in range(case["rows"]):
            b = int(case["perm"][i])
            for j in range(q_len):
                pos = case["cached"][i] + j
                page = case["own"][i][pos // case["tpb"]]
                expect[page, 0, :, pos % case["tpb"]] = case["k"][b * q_len + j]
                expect[page, 1, :, pos % case["tpb"]] = case["v"][b * q_len + j]
        assert torch.equal(pool, expect)


@needs_fa4
@pytest.mark.parametrize("q_len", [9])
def test_fa4_sampling_attention_graph_replay_is_bit_exact(q_len):
    """One CUDA graph of a layer, replayed on new inputs, equals eager calls on the same buffers bit
    for bit."""
    device = torch.device("cuda")
    num_heads, num_kv_heads, head_dim = 20, 20, 128
    case = _spec_case(num_heads, num_kv_heads, head_dim, q_len, True, 2, 7, True)
    prefix = _make_prefix(case, num_heads, head_dim, q_len)
    tail = _make_tail(case, num_heads, head_dim, q_len)
    q, k, v = case["q"].clone(), case["k"].clone(), case["v"].clone()
    pool = case["pool"].clone()
    out_graph = torch.zeros_like(q)
    _run(case, pool, q, k, v, out_graph, prefix, tail, head_dim)  # warm up / compile
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(case, pool, q, k, v, out_graph, prefix, tail, head_dim)
    gen = torch.Generator(device=device).manual_seed(11)
    for _ in range(3):
        q.copy_(torch.randn(q.shape, generator=gen, device=device).bfloat16())
        k.copy_(torch.randn(k.shape, generator=gen, device=device).bfloat16())
        v.copy_(torch.randn(v.shape, generator=gen, device=device).bfloat16())
        eager_pool = pool.clone()
        eager_out = torch.zeros_like(q)
        _run(case, eager_pool, q, k, v, eager_out, prefix, tail, head_dim)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_graph, eager_out)
        assert torch.equal(pool, eager_pool)


def test_fa4_prefix_splits_follow_the_grid():
    from tensorrt_llm._torch.attention.backends.fmha.sampling_shared_prefix_fa4 import (
        fa4_prefix_splits,
    )

    assert fa4_prefix_splits(120, 9, 28, 4, 132) == 1  # 60 tiles x 4 KV heads
    assert fa4_prefix_splits(60, 4, 28, 4, 132) == 3  # 14 tiles x 4 KV heads
    assert fa4_prefix_splits(4, 9, 28, 4, 132) == 8  # 2 tiles x 4 KV heads, capped
    assert fa4_prefix_splits(4, 9, 20, 20, 132) == 7  # MHA: 1 tile x 20 heads


# --------------------------------------------------------------------------- #
# End to end: EAGLE-3 n-sampling
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _accepted_drafts():
    """Counts the draft tokens the speculative worker accepts.

    ``LLM.get_stats`` returns nothing for this configuration, so the executor's per-iteration
    stats are read as they are produced; the executor must run in this process
    (``TLLM_WORKER_USE_SINGLE_PROCESS=1``).
    """
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    counts = {"accepted": 0, "requests": 0}
    original = PyExecutor._update_iter_stats

    def update(self, stats, *args, **kwargs):
        result = original(self, stats, *args, **kwargs)
        spec = getattr(stats, "specdec_stats", None)
        if spec is not None and spec.num_requests_with_draft_tokens > 0:
            counts["accepted"] += spec.num_accepted_tokens
            counts["requests"] += spec.num_requests_with_draft_tokens
        return result

    PyExecutor._update_iter_stats = update
    try:
        yield counts
    finally:
        PyExecutor._update_iter_stats = original


def _eagle3_paths():
    """(target, draft) checkpoints of the EAGLE-3 e2e tests, from the environment."""
    target = os.environ.get("TLLM_TEST_EAGLE3_TARGET")
    draft = os.environ.get("TLLM_TEST_EAGLE3_DRAFT")
    if target and draft and os.path.isdir(target) and os.path.isdir(draft):
        return target, draft
    return None


needs_eagle3 = pytest.mark.skipif(
    _eagle3_paths() is None,
    reason="set TLLM_TEST_EAGLE3_TARGET / TLLM_TEST_EAGLE3_DRAFT (e.g. Qwen2.5-7B-Instruct and its "
    "EAGLE-3 head)",
)
_LIBRARY_ENV = {
    "TLLM_FMHA_LIBS": f"+{lib.LIB_NAME}",
    lib.MIN_SHARED_ENV: "0",
    lib.MIN_GROUP_ENV: "2",
}


@contextlib.contextmanager
def _env(values):
    values = {"TLLM_WORKER_USE_SINGLE_PROCESS": "1", **values}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _eagle3_llm(prompts, n, max_tokens, draft_len, block_reuse=True):
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, Eagle3DecodingConfig, KvCacheConfig

    target, draft = _eagle3_paths()
    return LLM(
        model=target,
        max_batch_size=len(prompts) * n,
        max_num_tokens=8192,
        max_seq_len=max(len(p) for p in prompts) + max_tokens + draft_len + 1,
        speculative_config=Eagle3DecodingConfig(max_draft_len=draft_len, speculative_model=draft),
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=block_reuse,
            use_kv_cache_manager_v2=True,
            free_gpu_memory_fraction=0.3,
        ),
        cuda_graph_config=CudaGraphConfig(max_batch_size=len(prompts) * n),
        skip_tokenizer_init=True,
        enable_iter_perf_stats=True,
    )


@pytest.mark.skipif(not _fa4_available(), reason="needs SM90 and FlashAttention-4")
@needs_eagle3
@pytest.mark.threadleak(enabled=False)
def test_eagle3_n_sampling_with_library_keeps_acceptance_e2e():
    """EAGLE-3 (3 draft tokens) with n-sample requests, CUDA graphs on: with the library forced open
    every child produces its tokens and the mean accepted length stays within 5% of the run without
    the library."""
    from tensorrt_llm import SamplingParams

    prompts = [[1024 + ((29 * i + 7 * p) % 50000) for p in range(900 - 100 * i)] for i in range(2)]
    n, max_tokens = 8, 96

    def run(env):
        with _env(env):
            llm = _eagle3_llm(prompts, n, max_tokens, 3, block_reuse=False)
            with _accepted_drafts() as counts, llm:
                sampling = SamplingParams(
                    n=n, temperature=1.0, max_tokens=max_tokens, ignore_eos=True, end_id=-1
                )
                outputs = llm.generate(prompts, sampling_params=sampling)
        assert all(len(c.token_ids) == max_tokens for o in outputs for c in o.outputs)
        accepted, requests = counts["accepted"], counts["requests"]
        return (accepted + requests) / requests if requests else None

    off = run({})
    on = run(_LIBRARY_ENV)
    assert off is not None and on is not None, "no speculative-decoding stats reported"
    assert abs(on - off) <= 0.05 * off, (on, off)


@pytest.mark.skipif(not _fa4_available(), reason="needs SM90 and FlashAttention-4")
@needs_eagle3
@pytest.mark.threadleak(enabled=False)
def test_eagle3_greedy_n_sampling_matches_fa4_e2e():
    """Greedy EAGLE-3 (8 draft tokens) n-sample children with the library and with FlashAttention-4
    alone (the kernel the library takes the rows from) give the same first 16 tokens. The prompts
    repeat a 64-token pattern, which the model continues with a clear margin: on the regular kernel
    without speculation the top-2 logprob gap of every one of the first 16 greedy tokens is at least
    0.5 nats (output logprobs are not available with speculative decoding), so a bf16 rounding
    difference cannot flip them; random token prompts have exact ties there. Prompts end at and
    around page boundaries; Qwen2 fuses RoPE into the attention op, so the library's RoPE runs
    too."""
    from tensorrt_llm import SamplingParams

    def repeated(i, length):
        pattern = [1024 + ((977 * (i + 1) + 131 * t) % 30000) for t in range(64)]
        return (pattern * (length // 64 + 1))[:length]

    prompts = [repeated(i, length) for i, length in enumerate((511, 512, 513))]
    n, max_tokens = 6, 16

    def run(env):
        with _env({"TLLM_ALLOW_N_GREEDY_DECODING": "1", **env}):
            with _eagle3_llm(prompts, n, max_tokens, 8) as llm:
                sampling = SamplingParams(
                    n=n, top_k=1, max_tokens=max_tokens, ignore_eos=True, end_id=-1
                )
                outputs = llm.generate(prompts, sampling_params=sampling)
        return [[tuple(c.token_ids) for c in out.outputs] for out in outputs]

    reference = run({"TLLM_FMHA_LIBS": "+flash_attn4_generation"})
    candidate = run({**_LIBRARY_ENV, "TLLM_FMHA_LIBS": f"+{lib.LIB_NAME},+flash_attn4_generation"})
    for ref_samples, cand_samples in zip(reference, candidate):
        assert len(cand_samples) == n
        for ref_tokens, cand_tokens in zip(ref_samples, cand_samples):
            assert len(cand_tokens) == len(ref_tokens) == max_tokens
            assert cand_tokens == ref_tokens
