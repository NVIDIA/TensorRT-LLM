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
"""DSA metadata uploads staged into the step's input commit leave the copy path's device state.

While ``DSAtrtllmAttentionMetadata.prepare`` runs inside an open ``InputCommit`` (a CUDA-graph
generation step of the decode-step shell), its host-to-device uploads are staged and written by
the commit kernel, the block-offset clamps run on the host, and the Helix sparse-row gather runs
right after the commit. Each piece runs here on a metadata stand-in twice, once copying and once
staging (then committing), from the same poisoned buffers; the device buffers must end bitwise
equal, also over several steps with negative (padding) block ids and changing row layouts.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"


def _meta():
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )

    class _Meta(DSAtrtllmAttentionMetadata):
        pass

    return object.__new__(_Meta)


def _commit():
    from tensorrt_llm._torch.pyexecutor.input_commit import InputCommit

    return InputCommit(capacity=1 << 12)


def _run(md, fn, staged):
    """``fn()`` copying, or staged into a fresh commit that is then committed."""
    if not staged:
        fn()
    else:
        commit = _commit()
        commit.begin()
        md._staging_commit = commit
        try:
            fn()
        finally:
            md._staging_commit = None
        commit.commit()
    torch.cuda.synchronize()


def test_stage_or_copy_routes_rows_of_a_wider_buffer():
    md = _meta()
    gen = torch.Generator().manual_seed(0)
    host = torch.randint(-5, 1000, (3, 7), dtype=torch.int32, generator=gen)
    out = {}
    for staged in (False, True):
        buf = torch.full((4, 11), -9, dtype=torch.int32, device=DEV)
        _run(md, lambda: md._stage_or_copy(buf[:3, :7], host), staged)
        out[staged] = buf
    assert torch.equal(out[False], out[True])


def _block_offset_meta(num_seqs, max_seqs, width, seed):
    md = _meta()
    gen = torch.Generator().manual_seed(seed)
    pool = torch.randint(-1, 5000, (num_seqs, width - 2), dtype=torch.int32, generator=gen)
    md._seq_lens = torch.ones(num_seqs, dtype=torch.int32)
    md.request_ids = list(range(num_seqs))
    md._num_contexts = 0
    md.beam_width = 1
    md.kv_cache_manager = SimpleNamespace(
        get_pool_block_indices=lambda *a, **k: pool, indexer_k_cache_page_scale=2
    )
    md.host_indexer_k_cache_block_offsets = torch.full((max_seqs, width), -3, dtype=torch.int32)
    md.indexer_k_cache_block_offsets = torch.full(
        (max_seqs, width), -4, dtype=torch.int32, device=DEV
    )
    return md


def test_indexer_block_offsets_staged_match_the_copy_and_clamp():
    """Padding ids (-1) clamp to 0 on the host; rows the step leaves alone keep the copy path's
    values over several steps with fewer and more sequences."""
    max_seqs, width = 6, 16
    bufs = {}
    for staged in (False, True):
        buf = torch.full((max_seqs, width), -4, dtype=torch.int32, device=DEV)
        for step, num_seqs in enumerate((4, 2, 6, 1)):
            md = _block_offset_meta(num_seqs, max_seqs, width, seed=step)
            md.indexer_k_cache_block_offsets = buf
            if staged:
                md.__dict__["_staged_clamped_buffers"] = bufs.setdefault("clamped", set())
            _run(md, md._update_indexer_k_cache_block_offsets, staged)
        bufs[staged] = buf
    assert (bufs[False] >= 0).all()
    assert torch.equal(bufs[False], bufs[True])


def _helix_rows_meta(gen_lens, num_contexts=0, seed=0, pools=1, max_blocks=12):
    md = _meta()
    gen = torch.Generator().manual_seed(seed)
    num_seqs = num_contexts + len(gen_lens)
    rows_cap = 64
    md._num_contexts = num_contexts
    md._num_generations = len(gen_lens)
    md._seq_lens = torch.tensor([3] * num_contexts + list(gen_lens), dtype=torch.int32)
    md._seq_lens_kv = None
    md.num_sparse_topk = 2048
    md.prompt_lens_cpu = torch.full((num_seqs,), 5, dtype=torch.int32)
    md.helix_row_kv_lens_cuda = torch.full((rows_cap,), -1, dtype=torch.int32, device=DEV)
    md.helix_row_req_idx_cuda = torch.full((rows_cap,), -1, dtype=torch.long, device=DEV)
    md.helix_row_prompt_lens_cuda = torch.full((rows_cap,), -1, dtype=torch.int32, device=DEV)
    md.helix_row_kv_lens_host = torch.zeros(rows_cap, dtype=torch.int32)
    md.helix_row_req_idx_host = torch.zeros(rows_cap, dtype=torch.long)
    md.helix_row_prompt_lens_cpu = torch.zeros(rows_cap, dtype=torch.int32)
    md.helix_row_request_types_host = torch.ones(rows_cap, dtype=torch.int32)
    md.helix_row_block_offsets = torch.full(
        (pools, rows_cap, 2, max_blocks), -1, dtype=torch.int32, device=DEV
    )
    md.kv_cache_block_offsets = torch.randint(
        0, 1 << 20, (pools, num_seqs + 2, 2, max_blocks), dtype=torch.int32, generator=gen
    ).to(DEV)
    kv_lens = torch.randint(1, 100000, (num_seqs,), dtype=torch.int32, generator=gen)
    return md, kv_lens


@pytest.mark.parametrize("gen_lens", [(8,), (8, 8, 8, 8), (1, 1), (8, 4, 8)])
def test_helix_sparse_rows_staged_match_the_copy(gen_lens):
    names = (
        "helix_row_kv_lens_cuda",
        "helix_row_req_idx_cuda",
        "helix_row_prompt_lens_cuda",
        "helix_row_block_offsets",
    )
    out = {}
    for staged in (False, True):
        md, kv_lens = _helix_rows_meta(gen_lens, seed=len(gen_lens))
        for _ in range(2):  # staged: the gather runs after the commit, on the committed index
            _run(md, lambda: md._prepare_helix_sparse_rows(kv_lens), staged)
        out[staged] = {n: getattr(md, n).clone() for n in names}
        assert md._helix_num_rows == sum(gen_lens)
    for n in names:
        assert torch.equal(out[False][n], out[True][n]), n


def test_staged_prepare_leaves_the_schedules_to_the_step_refresh():
    """With a staging commit the decode schedules are not built in prepare(): they would read
    staged lengths before the commit, and the graph's on_update_kv_lens rebuilds them."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as indexer_mod

    md = _meta()
    md._num_contexts = 0
    md._num_generations = 2
    md._indexer_compress_ratio = 1
    md.kv_lens_cuda_runtime = torch.arange(1, 5, dtype=torch.int32, device=DEV)
    md._staging_commit = object()

    def forbidden(*args, **kwargs):
        raise AssertionError("DeepGEMM schedule built while staging")

    original = indexer_mod.get_paged_mqa_logits_metadata
    indexer_mod.get_paged_mqa_logits_metadata = forbidden
    try:
        indexer_mod.Indexer.prepare_scheduler_metadata(md)
    finally:
        indexer_mod.get_paged_mqa_logits_metadata = original
    assert torch.equal(md.gen_indexer_kv_lens_cuda_runtime, md.kv_lens_cuda_runtime[:2])


@pytest.mark.parametrize(
    "helix_spec, num_contexts, ragged, flushes",
    [
        (True, 0, False, False),  # the Helix speculative generation step
        (False, 0, False, True),
        (True, 1, False, True),
        (True, 0, True, True),
    ],
)
def test_staged_prepare_flushes_only_where_a_device_read_needs_it(
    monkeypatch, helix_spec, num_contexts, ragged, flushes
):
    """On a Helix speculative generation step every device read of a staged input runs after
    the commit, so prepare() stages into the open commit without flushing it first; other
    staged steps flush the inputs staged so far before the DSA preparation."""
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.pyexecutor.input_commit import input_commit_scope

    monkeypatch.setattr(TrtllmAttentionMetadata, "prepare", lambda self: None)
    md = _meta()
    md.enable_helix = True
    md._helix_spec_tokens_valid = helix_spec
    md._num_contexts = num_contexts
    md.ragged_verify_lens = [1] if ragged else None
    md.enable_ragged_verification = True
    seen = []
    md._prepare_dsa = lambda: seen.append(md._staging_commit)
    commit = _commit()
    staged = torch.zeros(4, dtype=torch.int32, device=DEV)
    commit.begin()
    with input_commit_scope(commit):
        commit.stage(staged, [1, 2, 3, 4])
        md.prepare()
    assert seen == [commit] and md._staging_commit is None and commit.is_open
    assert commit.num_commits == (1 if flushes else 0)
    commit.commit()
    torch.cuda.synchronize()
    assert staged.tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "seq_lens, spec",
    [((8,), True), ((8, 8, 8, 8), True), ((1, 1, 1), False)],
)
def test_helix_slot_mappings_staged_once_match_the_copy(monkeypatch, seq_lens, spec):
    """The Helix generation rows rewrite slot mappings the plain pass wrote. Staged, the
    overlapping rows go up once with their final values (two staged uploads of the same bytes
    race in the commit kernel); the device buffers equal the copy path's, under the commit
    verify (every commit checked)."""
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer, IndexerParams
    from tensorrt_llm._torch.pyexecutor import input_commit as input_commit_module

    monkeypatch.setattr(input_commit_module, "_VERIFY_COMMITS", -1)
    n = len(seq_lens)
    num_tokens = sum(seq_lens)
    tpb, head_dim, qbs = 64, 128, 128
    gen = torch.Generator().manual_seed(n)
    cached = torch.randint(0, 5 * tpb, (n,), generator=gen).tolist()
    params = IndexerParams(
        num_contexts=0,
        num_generations=n,
        num_ctx_tokens=0,
        head_dim=head_dim,
        quant_block_size=qbs,
        tokens_per_block=tpb,
        compress_ratio=1,
        request_ids=list(range(n)),
        num_past_tokens=cached,
        seq_lens=torch.tensor(seq_lens),
    )
    block_offsets = torch.randint(0, 1000, (n, 8), dtype=torch.int32, generator=gen)
    out = {}
    for staged in (False, True):
        md = _meta()
        md._num_contexts = 0
        md._num_generations = n
        md._seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        md._seq_lens_kv = None
        md._num_ctx_tokens = 0
        md._num_tokens = num_tokens
        md.enable_helix = True
        md._helix_spec_tokens_valid = spec
        md.helix_is_inactive_rank_cpu = torch.tensor([i % 2 == 1 for i in range(n)])
        md.host_indexer_k_cache_block_offsets = block_offsets
        md.host_slot_mapping_fp8 = torch.full((64,), -3, dtype=torch.int64)
        md.host_slot_mapping_scale = torch.full((64,), -3, dtype=torch.int64)
        md.slot_mapping_fp8 = torch.full((64,), -5, dtype=torch.int64, device=DEV)
        md.slot_mapping_scale = torch.full((64,), -5, dtype=torch.int64, device=DEV)
        _run(md, lambda: Indexer.recompute_slot_mappings(md, params), staged)
        out[staged] = (md.slot_mapping_fp8.clone(), md.slot_mapping_scale.clone())
    assert torch.equal(out[False][0], out[True][0])
    assert torch.equal(out[False][1], out[True][1])
