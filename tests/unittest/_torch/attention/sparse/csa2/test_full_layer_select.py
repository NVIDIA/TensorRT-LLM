# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Full layer's attention-TP decode selection chain selects, scores and publishes what the plain
chain does, eagerly and in replayed graphs."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
    CSA2ForwardState,
    CSA2Layout,
    CSA2Params,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows
from tensorrt_llm._torch.modules.top_k import TopKImplementation
from tensorrt_llm._utils import is_blackwell_datacenter
from tensorrt_llm.mapping import Mapping

from ._utils import _plan_indexer
from .test_indexer import _projected, _StridedOwner

ROWS, TOP_K = 6, 512
# The decode graphs' arena geometry: wider than the one-launch decode select and the finalize's bitmap.
WIDTH = 1 << 19

skip_unless_tp_chain = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_blackwell_datacenter(),
    reason="Attention-TP selection chain runs on SM100 / SM103",
)


def _full_layer_case(ratio: int, visible_end: int, width: int = WIDTH):
    """One request decoding ``ROWS`` rows on Full layer 0 (a candidate source at ratio 1)."""
    layout = CSA2Layout(
        (ratio,),
        (0,),
        (0,),
        candidate_source_layer_id=0 if ratio == 1 else None,
        index_topk=TOP_K,
    )
    page = 128 // ratio
    used = -(-visible_end // page)
    manager = _StridedOwner(
        layout,
        pack_rows(torch.randn(used * page, 128, device="cuda", dtype=torch.bfloat16), "index"),
    )
    manager.max_total_draft_tokens = ROWS - 1
    q, _, weights = _projected(ROWS)
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=ROWS)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping()
    metadata.is_cuda_graph = True
    metadata.csa2_kv_sources = {0: 0}
    metadata.csa2_num_context_requests = 0
    metadata.csa2_request_query_ranges = [(0, ROWS)]
    metadata.csa2_request_start_positions = [visible_end * ratio - ROWS]
    metadata.csa2_request_lengths = [ROWS]
    metadata.csa2_global_max_positions = {0: width}
    metadata.csa2_global_page_sizes = {0: page}
    table = torch.full((1, -(-width // page)), -1, dtype=torch.int32, device="cuda")
    table[0, :used] = torch.randperm(used, device="cuda").to(torch.int32)
    table[0, used // 3] = -1  # A hole inside the visible prefix.
    visible = torch.arange(
        visible_end - ROWS + 1, visible_end + 1, device="cuda", dtype=torch.int32
    )
    metadata.csa2_global_page_tables = {0: table}
    metadata.csa2_visible_lengths = {0: visible}
    metadata.csa2_token_requests = torch.zeros(ROWS, device="cuda", dtype=torch.int64)
    metadata.csa2_decode_row_requests = torch.zeros(ROWS, device="cuda", dtype=torch.int32)
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.csa2_candidate_blocks = {}
    metadata.csa2_candidate_counts = {}
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = -1
    _plan_indexer(metadata)
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(ROWS, 512, device="cuda", dtype=torch.bfloat16),
        index_q=q,
        index_weights=weights,
    )
    indexer = CSA2Indexer(layout, 0, 32, 128, CSA2Params(layout=layout, enable_heuristic_topk=True))
    return indexer, state, q, visible, table


def _publication(metadata):
    """The candidate source's publication buffers of the last call (none below ratio 1)."""
    return [
        tensor
        for tensor in (
            metadata.csa2_candidates.get(0),
            metadata.csa2_candidate_blocks.get(0),
            metadata.csa2_candidate_counts.get(0),
        )
        if tensor is not None
    ]


def _plain_chain(patch):
    """Take the selection chain of attention DP and other GPUs."""
    patch.setattr(CSA2Indexer, "_tp_selection_chain", staticmethod(lambda metadata: False))


@skip_unless_tp_chain
# Ratio 1: a candidate source, whose publication the chain must not change.
@pytest.mark.parametrize("ratio, width", [(1, WIDTH - 40)])
@torch.inference_mode()
def test_tp_selection_chain_equals_plain_chain(monkeypatch, ratio, width):
    """The TP chain (page mask, 4-CTA cluster GVR with the folded finalize) matches the plain
    chain."""
    torch.manual_seed(4500 + ratio)
    indexer, state, q, visible, table = _full_layer_case(ratio, 9000 // ratio, width)
    assert indexer.top_k.decode_implementation == TopKImplementation.CUTE_DSL_GVR
    metadata = state.metadata
    select = CSA2Indexer._select_paged_logits
    select_scores = CSA2Indexer.select_prepared_scores
    logits_seen, folded, published = [], [], []

    def recording_select(self, metadata, logits, *args, **kwargs):
        out = select(self, metadata, logits, *args, **kwargs)
        if not torch.cuda.is_current_stream_capturing():
            logits_seen.append(logits.clone())
        return out

    def recording_scores(self, *args, **kwargs):
        out = select_scores(self, *args, **kwargs)
        fold = kwargs.get("gvr_finalize")
        if not kwargs.get("candidate", False) and not kwargs.get("is_prefill", True):
            folded.append(fold is not None and fold["done"])
        return out

    monkeypatch.setattr(CSA2Indexer, "_select_paged_logits", recording_select)
    monkeypatch.setattr(CSA2Indexer, "select_prepared_scores", recording_scores)

    def both():
        published.clear()
        metadata._csa2_forward_serial = getattr(metadata, "_csa2_forward_serial", 0) + 1
        outs = []
        for tp_chain in (False, True):
            with monkeypatch.context() as patch:
                if not tp_chain:
                    _plain_chain(patch)
                outs.append(indexer(state, 0, ROWS))
                metadata.wait_candidates(0)
            published.append(_publication(metadata))
        return outs

    def same_publication():
        return all(torch.equal(a, b) for a, b in zip(*published))

    for _ in range(2):
        folded.clear()
        logits_seen.clear()
        plain, chain = both()
        assert folded == [False, True]
        assert torch.equal(chain, plain)
        assert same_publication()
        assert int((plain >= 0).sum()) == ROWS * TOP_K
        context = int(visible[-1])
        assert logits_seen[0].stride() == logits_seen[1].stride()
        assert torch.equal(
            logits_seen[0][:, :context].view(torch.int32),
            logits_seen[1][:, :context].view(torch.int32),
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plain, chain = both()
    for trial in range(4):
        q.neg_() if trial % 2 else q.mul_(0.5)
        if trial == 2:
            visible.sub_(1000 // ratio)
        if trial == 3:
            table[0, 1] = -1
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(chain, plain), trial
        assert same_publication(), trial
        assert int((plain >= 0).sum()) == ROWS * TOP_K, trial
