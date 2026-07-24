# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for SAWorker promoting accepted recurrent states on
hybrid (SSM) models.

Verification writes per-step recurrent states to the cache manager's
speculative scratch buffers, never the live pools; the worker must promote
the accepted step via ``update_mamba_states`` after acceptance (mirroring
the dflash/eagle3 one-engine workers), or hybrid models silently corrupt
their recurrent state under standalone SA speculative decoding.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MixedMambaHybridCacheManager
from tensorrt_llm._torch.speculative.sa_worker import SAWorker


def _make_worker() -> SAWorker:
    spec_config = SimpleNamespace(max_draft_len=4, max_matching_ngram_size=2)
    worker = SAWorker(spec_config)
    # Stub out everything around the state-promotion call site; the mocked
    # sampler return values flow into update_mamba_states unchanged.
    worker._execute_guided_decoder_if_present = mock.MagicMock()
    worker._sample_and_accept_draft_tokens = mock.MagicMock(
        return_value=(mock.sentinel.accepted_tokens, mock.sentinel.num_accepted_tokens)
    )
    worker._generate_draft_tokens = mock.MagicMock(return_value=mock.sentinel.next_draft_tokens)
    worker._prepare_next_new_tokens = mock.MagicMock(return_value=mock.sentinel.next_new_tokens)
    return worker


def _make_metadata(kv_cache_manager, num_seqs: int, num_contexts: int):
    attn_metadata = SimpleNamespace(
        num_seqs=num_seqs,
        num_contexts=num_contexts,
        kv_cache_manager=kv_cache_manager,
        mamba_metadata=SimpleNamespace(state_indices=mock.sentinel.state_indices),
    )
    spec_metadata = SimpleNamespace(
        runtime_draft_len=4,
        batch_indices_cuda=mock.sentinel.batch_indices_cuda,
    )
    return attn_metadata, spec_metadata


def _run_forward(worker, attn_metadata, spec_metadata):
    return worker._forward_impl(
        input_ids=mock.sentinel.input_ids,
        position_ids=mock.sentinel.position_ids,
        hidden_states=mock.sentinel.hidden_states,
        logits=mock.sentinel.logits,
        attn_metadata=attn_metadata,
        spec_metadata=spec_metadata,
    )


def test_hybrid_manager_promotes_accepted_states():
    worker = _make_worker()
    manager = mock.MagicMock(spec=MixedMambaHybridCacheManager)
    attn_metadata, spec_metadata = _make_metadata(manager, num_seqs=2, num_contexts=0)

    result = _run_forward(worker, attn_metadata, spec_metadata)

    manager.update_mamba_states.assert_called_once_with(
        attn_metadata=attn_metadata,
        num_accepted_tokens=mock.sentinel.num_accepted_tokens,
        state_indices=mock.sentinel.state_indices,
    )
    assert result["new_tokens"] is mock.sentinel.accepted_tokens


def test_hybrid_manager_context_only_batch_skips_promotion():
    worker = _make_worker()
    manager = mock.MagicMock(spec=MixedMambaHybridCacheManager)
    attn_metadata, spec_metadata = _make_metadata(manager, num_seqs=2, num_contexts=2)

    _run_forward(worker, attn_metadata, spec_metadata)

    manager.update_mamba_states.assert_not_called()


def test_pure_attention_manager_skips_promotion():
    worker = _make_worker()
    manager = mock.MagicMock()  # not a MambaHybridCacheManager
    attn_metadata, spec_metadata = _make_metadata(manager, num_seqs=2, num_contexts=0)

    _run_forward(worker, attn_metadata, spec_metadata)

    manager.update_mamba_states.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
