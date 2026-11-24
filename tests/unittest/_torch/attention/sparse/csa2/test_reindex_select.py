# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Reindex layer's decode candidate logits and mapped Top-K in one launch (``reindex_select.py``) select
exactly what DeepGEMM's paged sparse MQA logits followed by ``select_mapped_consumer_topk_`` select."""

import pytest
import torch
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.attention.backends.sparse.csa2 import indexer as indexer_module
from tensorrt_llm._torch.attention.backends.sparse.csa2 import reindex_select as rs
from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2ForwardState
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows
from tensorrt_llm._torch.modules.top_k import TopKImplementation
from tensorrt_llm._utils import is_blackwell_datacenter
from tensorrt_llm.mapping import Mapping

from ._utils import _plan_indexer
from .test_indexer import (
    _projected,
    _sparse_candidate_layout,
    _sparse_kernels_available,
    _StridedOwner,
)

NBLK, SB, TOP_K = 2048, 8, 512


@skip_pre_blackwell
@pytest.mark.skipif(
    not _sparse_kernels_available() or not is_blackwell_datacenter(),
    reason="DeepGEMM sparse MQA and the fused select on SM100 / SM103",
)
@torch.inference_mode()
def test_indexer_consumer_fused_equals_two_launches(monkeypatch):
    """A Reindex layer's decode in the fused launch, its source skipping the sparse logits schedule,
    selects what two launches select, eagerly and in graphs."""
    torch.manual_seed(4402)
    requests, tokens, page = 2, 6, 128
    next_n = tokens // requests
    width = 24576
    layout = _sparse_candidate_layout(NBLK, SB, topk=TOP_K)
    manager = _StridedOwner(
        layout, pack_rows(torch.randn(width, 128, device="cuda", dtype=torch.bfloat16), "index")
    )
    manager.max_total_draft_tokens = next_n - 1
    q, _, weights = _projected(tokens)
    metadata = CSA2TrtllmMetadata(max_num_requests=requests, max_num_tokens=tokens)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping()
    metadata.is_cuda_graph = True
    metadata.csa2_kv_sources = {0: 0, 1: 0}
    metadata.csa2_num_context_requests = 0
    metadata.csa2_request_query_ranges = [(r * next_n, (r + 1) * next_n) for r in range(requests)]
    metadata.csa2_request_start_positions = [width - next_n - 1000 * r for r in range(requests)]
    metadata.csa2_request_lengths = [next_n] * requests
    metadata.csa2_global_max_positions = {0: width}
    metadata.csa2_global_page_sizes = {0: page}
    table = torch.stack([torch.randperm(width // page, device="cuda") for _ in range(requests)]).to(
        torch.int32
    )
    rows = torch.arange(requests, device="cuda").repeat_interleave(next_n)
    visible = torch.tensor(
        [width - 1000 * r - next_n + i + 1 for r in range(requests) for i in range(next_n)],
        device="cuda",
        dtype=torch.int32,
    )
    metadata.csa2_global_page_tables = {0: table}
    metadata.csa2_visible_lengths = {0: visible, 1: visible}
    metadata.csa2_token_requests = rows.to(torch.int64)
    metadata.csa2_decode_row_requests = rows.to(torch.int32)
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.csa2_candidate_blocks = {}
    metadata.csa2_candidate_counts = {}
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = -1
    _plan_indexer(metadata)
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(tokens, 512, device="cuda", dtype=torch.bfloat16),
        index_q=q,
        index_weights=weights,
    )
    source = CSA2Indexer(layout, 0, 32, 128)
    consumer = CSA2Indexer(layout, 1, 32, 128)
    assert consumer.top_k.decode_implementation == TopKImplementation.CUTE_DSL_RADIX
    calls, prebuilds = [], []
    original = rs.reindex_select_

    def counting(*args, **kwargs):
        calls.append(args[0].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(rs, "reindex_select_", counting)
    prebuild = metadata.prebuild_sparse_schedule

    def counting_prebuild(*args, **kwargs):
        prebuilds.append(args[0])
        return prebuild(*args, **kwargs)

    monkeypatch.setattr(metadata, "prebuild_sparse_schedule", counting_prebuild)
    # Each path records its own consumers' use of the sparse logits schedule.
    schedule_needed = {False: {}, True: {}}

    def forward(fused):
        """The source publishes, then the consumer selects with the fused launch or two launches."""
        with monkeypatch.context() as patch:
            patch.setattr(indexer_module, "_SCHEDULE_NEEDED", schedule_needed[fused])
            if not fused:
                patch.setattr(rs, "_FUSED_SELECT_MAX_ROWS", 0)
            metadata._csa2_forward_serial = getattr(metadata, "_csa2_forward_serial", 0) + 1
            source(state, 0, tokens)
            return consumer(state, 0, tokens)

    def both():
        return forward(False), forward(True)

    # The first forward records the consumer's fused select; later ones skip the prebuild.
    forward(True)
    for _ in range(2):
        calls.clear()
        prebuilds.clear()
        plain, fused = both()
        assert calls == [tokens]
        assert prebuilds == [0]
        assert torch.equal(fused, plain)
        assert int((plain >= 0).sum()) == tokens * TOP_K
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plain, fused = both()
    for trial in range(3):
        q.neg_() if trial % 2 else q.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(fused, plain), trial
        assert int((plain >= 0).sum()) == tokens * TOP_K, trial
