#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused invariant test for the Inkling attention decode-metadata publish.

The Inkling Triton decode kernel needs, per generation request, the total KV
length (``num_cached + 1``) and the physical page table. The runtime used to
build these from host lists INSIDE ``model.forward``
(``torch.tensor(..., device=cuda)`` + ``build_page_table``), which raises
``Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the
CPU tensor is pinned`` under the enabled ``cuda_graph=true, overlap=true`` config
(job 5460908: rank-0 traceback at ``modeling_inkling.py`` ``_run_generation``).

``InklingDecodeMeta`` fixes that by publishing this batch's decode metadata into
per-layer STABLE GPU buffers EAGERLY (before capture/replay), so the captured
forward reads them with no host->device copy -- the same stable-pointer contract
as the short-conv ``state_indices`` pool. This test pins that behavior on CPU (no
checkpoint / no GPU / no full attention module needed -- it is pure buffer
bookkeeping):

* first refresh allocates buffers, marks ready, and writes the correct
  ``num_cached + 1`` seq lens and padded page table;
* a same-or-smaller batch REUSES the buffers (stable ``data_ptr``) and only
  overwrites their contents -- the invariant a captured graph depends on;
* an oversubscribing batch grows the row capacity elastically (outside capture);
* growth is REFUSED under CUDA graph (``is_cuda_graph=True``) -- growing would
  strand the captured pointer;
* a context-only batch (no generation slice) is a no-op that leaves ``ready``
  False.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.models.modeling_inkling import InklingDecodeMeta


def _fake_mgr(block_map, max_blocks_per_seq=8):
    """Duck-typed KVCacheManagerV2: get_batch_cache_indices(req_ids, layer) ->
    per-request physical page lists (from ``block_map``); max_blocks_per_seq is
    the fixed page-table width bound."""

    def get_batch_cache_indices(request_ids, layer_idx):
        return [list(block_map[r]) for r in request_ids]

    return SimpleNamespace(get_batch_cache_indices=get_batch_cache_indices,
                           max_blocks_per_seq=max_blocks_per_seq)


def _fake_md(mgr, request_ids, num_cached, num_contexts=0, is_cuda_graph=False):
    return SimpleNamespace(
        request_ids=list(request_ids),
        num_contexts=num_contexts,
        kv_cache_manager=mgr,
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=list(num_cached)),
        is_cuda_graph=is_cuda_graph,
    )


def test_decode_meta_publish_and_stable_pointer():
    dev = torch.device("cpu")
    # req 10 owns pages [4,5]; req 11 owns page [7]; req 12 owns pages [1,2,3].
    block_map = {10: [4, 5], 11: [7], 12: [1, 2, 3]}
    mgr = _fake_mgr(block_map, max_blocks_per_seq=8)
    meta = InklingDecodeMeta(layer_idx=3)
    assert meta.ready is False and meta.seq_lens is None

    # 1) First refresh: 2 generation requests, each with num_cached tokens.
    md = _fake_md(mgr, request_ids=[10, 11], num_cached=[130, 5])
    assert meta.refresh(md, dev) is True
    assert meta.ready is True
    assert meta.max_pages == 8
    assert meta.cap == 2
    # total-KV length = num_cached + 1.
    assert meta.seq_lens[:2].tolist() == [131, 6]
    # page table padded to max_pages, row i = req i's page list.
    assert meta.page_table[:2, :2].tolist() == [[4, 5], [7, 0]]
    assert meta.page_table.shape == (2, 8)

    sl_ptr = meta.seq_lens.data_ptr()
    pt_ptr = meta.page_table.data_ptr()

    # 2) Same-size batch (different requests/contents): buffers REUSED (stable
    #    pointer -- the captured graph reads the same address), contents updated.
    md2 = _fake_md(mgr, request_ids=[12, 11], num_cached=[200, 9])
    assert meta.refresh(md2, dev) is True
    assert meta.seq_lens.data_ptr() == sl_ptr
    assert meta.page_table.data_ptr() == pt_ptr
    assert meta.seq_lens[:2].tolist() == [201, 10]
    assert meta.page_table[:2, :3].tolist() == [[1, 2, 3], [7, 0, 0]]

    # 3) Smaller batch reuses the buffers too (cap only grows).
    md3 = _fake_md(mgr, request_ids=[11], num_cached=[3])
    assert meta.refresh(md3, dev) is True
    assert meta.cap == 2
    assert meta.seq_lens.data_ptr() == sl_ptr
    assert meta.seq_lens[:1].tolist() == [4]

    # 4) Oversubscribing batch grows the capacity (outside CUDA graph).
    md4 = _fake_md(mgr, request_ids=[10, 11, 12], num_cached=[1, 2, 3])
    assert meta.refresh(md4, dev) is True
    assert meta.cap == 3
    assert meta.seq_lens[:3].tolist() == [2, 3, 4]
    assert meta.page_table[:3, :3].tolist() == [[4, 5, 0], [7, 0, 0], [1, 2, 3]]


def test_decode_meta_growth_forbidden_under_cuda_graph():
    dev = torch.device("cpu")
    block_map = {10: [4, 5], 11: [7], 12: [1], 13: [2]}
    mgr = _fake_mgr(block_map, max_blocks_per_seq=8)
    meta = InklingDecodeMeta(layer_idx=0)

    # Prime at capacity 2 under capture (allocation is allowed on first touch).
    md = _fake_md(mgr, request_ids=[10, 11], num_cached=[10, 20],
                  is_cuda_graph=True)
    assert meta.refresh(md, dev) is True
    assert meta.cap == 2

    # A larger batch WHILE capturing/replaying would reallocate the stable
    # buffers and strand the captured pointer: refuse loudly.
    md_big = _fake_md(mgr, request_ids=[10, 11, 12, 13], num_cached=[1, 2, 3, 4],
                      is_cuda_graph=True)
    with pytest.raises(RuntimeError, match="during CUDA graph"):
        meta.refresh(md_big, dev)


def test_decode_meta_context_only_is_noop():
    dev = torch.device("cpu")
    mgr = _fake_mgr({10: [4]}, max_blocks_per_seq=8)
    meta = InklingDecodeMeta(layer_idx=1)
    # num_contexts == num_seqs -> no generation slice -> nothing published.
    md = _fake_md(mgr, request_ids=[10], num_cached=[0], num_contexts=1)
    assert meta.refresh(md, dev) is False
    assert meta.ready is False
    assert meta.seq_lens is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
