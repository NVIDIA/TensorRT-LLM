# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the real MTPSpecMetadata preparation/population paths.

These construct ACTUAL ``MTPSpecMetadata`` objects (vanilla MTP, rejection
enabled) and exercise the real preparation/population control flow -- no model
forward required:

  - ``test_mtp_prepare_allocates_rejection_buffers_*``: shared rejection buffers
    (``draft_probs`` / ``batch_slot_ids`` / ``full_draft_probs``) are allocated
    via the shared ``prepare_rejection_sampling_buffers`` contract with the
    expected shapes, dtypes, and CUDA placement, while the always-present
    MTP-specific buffers (``batch_indices_cuda`` / ``draft_token_indices_cuda``)
    stay intact; disabled-rejection leaves the large buffers None.
  - ``test_mtp_prepare_populates_hidden_state_buffers``: with a real
    ``MTPHiddenStatesManager``, ``prepare()`` populates ``slot_ids``,
    ``mtp_hidden_states_ptrs``, and ``mtp_past_tokens_ptrs`` consistently with
    the manager pools' ``data_ptr()``.
  - ``test_populate_sampling_params_copies_py_seq_slot``: the
    ``populate_sampling_params_for_one_model`` path copies non-contiguous
    per-request ``py_seq_slot`` values into ``batch_slot_ids``; the
    disabled-rejection case allocates no slot table.

Requires CUDA (the buffers/pools are CUDA tensors).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import SpeculativeDecodingMode
from tensorrt_llm._torch.speculative.mtp import MTPHiddenStatesManager, MTPSpecMetadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="MTPSpecMetadata buffers are CUDA tensors"
)

R, K, V, H = 8, 4, 32, 16  # max_num_requests, max_draft_len, vocab, hidden_size


def _make_metadata(use_rejection: bool, manager=None, num_generations: int = 2) -> MTPSpecMetadata:
    m = MTPSpecMetadata(
        max_num_requests=R,
        max_draft_len=K,
        max_total_draft_tokens=K,
        spec_dec_mode=SpeculativeDecodingMode.MTP,
        mtp_num_modules=K,
        mtp_hidden_states_manager=manager,
        use_rejection_sampling=use_rejection,
        vocab_size=V,
        num_generations=num_generations,
        num_tokens=10,
    )
    m.request_ids = [0, 1, 2, 3]
    return m


def test_mtp_prepare_allocates_rejection_buffers_and_keeps_mtp_buffers():
    m = _make_metadata(use_rejection=True)
    # MTP-specific buffers exist right after construction (__post_init__).
    assert m.batch_indices_cuda is not None
    assert m.draft_token_indices_cuda is not None
    m.prepare()
    # Shared rejection buffers now allocated with the expected shapes. The
    # slot-indexed prob buffers reserve one extra "scratch" row at index R
    # (R+1 rows total) for CUDA-graph dummy/padding requests whose py_seq_slot
    # is None, so their prob scatter never aliases a real request's slot.
    assert m.draft_probs is not None and tuple(m.draft_probs.shape) == (R + 1, K, V)
    assert m.batch_slot_ids is not None and m.batch_slot_ids.shape[0] == R
    assert m.full_draft_probs is not None and tuple(m.full_draft_probs.shape) == (R + 1, K, V)
    assert m.draft_probs_vocab_size == V
    # ...with the expected dtypes and CUDA placement (full buffer contract).
    assert m.draft_probs.dtype == torch.float32 and m.draft_probs.is_cuda
    assert m.full_draft_probs.dtype == torch.float32 and m.full_draft_probs.is_cuda
    assert m.batch_slot_ids.dtype == torch.long and m.batch_slot_ids.is_cuda
    # MTP-specific buffers are intact (not clobbered by the shared allocation).
    assert m.batch_indices_cuda is not None and m.batch_indices_cuda.shape[0] == R
    assert m.draft_token_indices_cuda is not None and m.draft_token_indices_cuda.shape[0] == K


def test_mtp_prepare_noop_rejection_buffers_when_disabled():
    m = _make_metadata(use_rejection=False)
    m.prepare()
    assert m.draft_probs is None
    assert m.batch_slot_ids is None
    assert m.full_draft_probs is None
    # MTP buffers still allocated regardless of rejection.
    assert m.batch_indices_cuda is not None
    assert m.draft_token_indices_cuda is not None


def test_mtp_prepare_populates_hidden_state_buffers():
    """Exercise the real MTPHiddenStatesManager hidden-state-pointer branch."""
    config = SimpleNamespace(max_draft_len=K, use_relaxed_acceptance_for_thinking=False)
    manager = MTPHiddenStatesManager(
        config=config, dtype=torch.float16, hidden_size=H, max_num_requests=R
    )
    request_ids = [0, 1, 2, 3]
    manager.add_dummy_requests(request_ids)

    m = _make_metadata(use_rejection=True, manager=manager)
    m.request_ids = request_ids
    # __post_init__ allocates the hidden-state pointer buffers when a manager
    # is attached.
    assert m.slot_ids is not None
    assert m.mtp_hidden_states_ptrs is not None
    assert m.mtp_past_tokens_ptrs is not None

    m.prepare()

    n = len(request_ids)
    # slot_ids are populated from the manager's slot table.
    expected_slots = [manager.slot_manager.get_slot(rid) for rid in request_ids]
    assert m.slot_ids[:n].cpu().tolist() == expected_slots
    # The pointer tables match the manager pools' data_ptr() for each slot.
    expected_hs_ptrs = [manager.mtp_past_hidden_states_pool[s].data_ptr() for s in expected_slots]
    expected_tok_ptrs = [manager.mtp_past_tokens_pool[s].data_ptr() for s in expected_slots]
    assert m.mtp_hidden_states_ptrs[:n].cpu().tolist() == expected_hs_ptrs
    assert m.mtp_past_tokens_ptrs[:n].cpu().tolist() == expected_tok_ptrs
    # Shared rejection buffers still allocated alongside the MTP pointers.
    assert m.draft_probs is not None and m.batch_slot_ids is not None


def _request(py_seq_slot: int):
    """A lightweight LlmRequest stand-in for the population scan."""
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

    return SimpleNamespace(
        sampling_config=SimpleNamespace(temperature=None, top_k=None, top_p=None),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=py_seq_slot,
    )


def test_populate_sampling_params_copies_py_seq_slot():
    """batch_slot_ids is filled from non-contiguous per-request py_seq_slot."""
    m = _make_metadata(use_rejection=True)
    m.runtime_draft_len = K
    slots = [5, 2, 7]
    m.populate_sampling_params_for_one_model([_request(s) for s in slots])
    # populate_* allocates batch_slot_ids via the idempotent helper, then copies
    # py_seq_slot into it -- proving the population path needs no model forward.
    assert m.batch_slot_ids is not None
    assert m.batch_slot_ids[: len(slots)].cpu().tolist() == slots


def test_populate_sampling_params_no_slot_table_when_disabled():
    m = _make_metadata(use_rejection=False)
    m.runtime_draft_len = K
    m.populate_sampling_params_for_one_model([_request(s) for s in [5, 2, 7]])
    assert m.batch_slot_ids is None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
