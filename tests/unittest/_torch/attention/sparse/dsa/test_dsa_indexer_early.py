# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The early DSA Top-K hand-off (MLA._dsa_indexer_early): the backend returns indices the MLA module predicted
ahead of the attention call for generation-only calls and predicts itself otherwise."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import DSABackendForwardArgs
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import DSATrtllmAttention


class _Indexer:
    def __init__(self):
        self.calls = 0

    def forward_from_projected(self, metadata, q, intermediates, is_generation=None):
        self.calls += 1
        raise RuntimeError("predicted although precomputed indices were handed over")


def test_precomputed_indices_short_circuit_generation():
    indices = torch.arange(12, dtype=torch.int32).view(3, 4)
    args = DSABackendForwardArgs(indexer_intermediates=[], precomputed_attn_indices=indices)
    backend = SimpleNamespace(indexer=_Indexer())
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only, sparse_backend_args=args
    )
    metadata = SimpleNamespace(num_ctx_tokens=0, num_tokens=3, shared_topk_indices=None)
    got, offsets = DSATrtllmAttention.sparse_attn_predict(
        backend, torch.empty(3, 8), None, metadata, forward_args
    )
    assert got is indices and offsets is None
    assert backend.indexer.calls == 0


def test_precomputed_indices_ignored_outside_generation():
    args = DSABackendForwardArgs(
        indexer_intermediates=[], precomputed_attn_indices=torch.zeros(2, 4, dtype=torch.int32)
    )
    backend = SimpleNamespace(indexer=_Indexer())
    forward_args = AttentionForwardArgs(
        attention_input_type=AttentionInputType.context_only, sparse_backend_args=args
    )
    metadata = SimpleNamespace(num_ctx_tokens=2, num_tokens=2, shared_topk_indices=None)
    with pytest.raises(RuntimeError, match="predicted"):
        DSATrtllmAttention.sparse_attn_predict(
            backend, torch.empty(2, 8), None, metadata, forward_args
        )
