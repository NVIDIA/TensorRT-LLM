# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding state that is keyed by live-request identity must be
sized by the sequence-slot pool, not by max_batch_size.

Under the attention-DP overlap headroom the two differ by 2x
(``compute_max_num_sequences``): a finished request holds its slot for one more
iteration while its replacement is already admitted (nvbug-6627795). Two
distinct families follow from that, and only the first needs the pool size:

* keyed by ``py_seq_slot`` / a per-request ``SlotManager`` slot -- must span the
  pool. ``SpecMetadata.num_seq_slots`` (draft_probs, full_draft_probs,
  penalty_state) and ``MTPHiddenStatesManager``'s hidden-state pools.
* keyed by *batch position* -- ``max_num_requests`` is correct and deliberately
  unchanged, because the micro-batch scheduler caps every forward at
  max_batch_size (``no_schedule_after_state=GENERATION_TO_COMPLETE`` also keeps
  the retiring requests out of the batch entirely).
"""

import inspect
import types

import pytest
import torch

from tensorrt_llm._torch.speculative.mtp import MTPHiddenStatesManager
from tensorrt_llm._torch.speculative.utils import _build_spec_metadata, get_spec_metadata

R, POOL = 8, 16  # max_batch_size, 2 * max_batch_size (overlap headroom)


@pytest.mark.cpu_only
def test_slot_pool_size_is_applied_centrally(monkeypatch):
    """``get_spec_metadata`` stamps the pool size onto whatever mode was built.

    This is the property that fixes the review finding: previously only the
    MTP-eagle branch forwarded ``num_seq_slots``, so vanilla MTP, Eagle3
    one-model, PARD, DFlash/DSpark and draft-target one-model all sized their
    slot-indexed buffers at ``max_num_requests``. Applying it once at the single
    exit point makes it impossible for a mode -- including a future one -- to be
    missed, so the assertion deliberately does not name any mode.
    """
    built = types.SimpleNamespace()
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.utils._build_spec_metadata", lambda *a, **k: built
    )
    spec_config = types.SimpleNamespace(enable_penalty=False)

    out = get_spec_metadata(
        spec_config,
        model_config=object(),
        max_num_requests=R,
        max_num_tokens=128,
        num_seq_slots=POOL,
    )

    assert out is built
    assert out.num_seq_slots == POOL


@pytest.mark.cpu_only
def test_unknown_slot_pool_leaves_the_max_num_requests_fallback(monkeypatch):
    """``num_seq_slots=None`` must not be written as a literal.

    Both allocators resolve the pool as ``self.num_seq_slots or
    self.max_num_requests``, so leaving the dataclass default (0) in place is how
    a caller that does not know the pool size keeps the old sizing. Writing
    ``None`` would raise in the ``+ 1`` scratch-row arithmetic instead.
    """
    built = types.SimpleNamespace()
    monkeypatch.setattr(
        "tensorrt_llm._torch.speculative.utils._build_spec_metadata", lambda *a, **k: built
    )
    spec_config = types.SimpleNamespace(enable_penalty=False)

    get_spec_metadata(
        spec_config,
        model_config=object(),
        max_num_requests=R,
        max_num_tokens=128,
        num_seq_slots=None,
    )

    assert not hasattr(built, "num_seq_slots")


@pytest.mark.cpu_only
def test_per_mode_builder_does_not_take_the_pool_size():
    """Guard the central-application invariant structurally.

    Re-plumbing ``num_seq_slots`` through the per-mode constructors is what let a
    branch be forgotten in the first place; keep the builder free of it.
    """
    assert "num_seq_slots" not in inspect.signature(_build_spec_metadata).parameters


def _mtp_config():
    return types.SimpleNamespace(max_draft_len=2, use_relaxed_acceptance_for_thinking=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MTP hidden-state pools are CUDA tensors")
@pytest.mark.parametrize(
    "num_seq_slots,expected_pool",
    [
        (POOL, POOL + 1),
        (None, R + 1),
    ],
)
def test_mtp_hidden_states_pool_spans_the_slot_pool(num_seq_slots, expected_pool):
    """The pool must cover every *resident* request, plus the CUDA-graph dummy.

    ``add_slot`` runs on a request's first context chunk and the slot is only
    returned by ``free_resources``, which the overlap scheduler defers -- so at
    ``max_num_requests + 1`` the replacement request raises ``NoFreeSlotsError``.
    ``None`` keeps the pre-existing sizing for callers that do not know the pool.
    """
    mgr = MTPHiddenStatesManager(
        _mtp_config(), torch.float16, hidden_size=8, max_num_requests=R, num_seq_slots=num_seq_slots
    )

    assert mgr.slot_manager.max_num_requests == expected_pool
    assert mgr.mtp_past_hidden_states_pool.shape[0] == expected_pool
    assert mgr.mtp_past_tokens_pool.shape[0] == expected_pool
    assert mgr.mtp_relaxed_delta_pool.shape[0] == expected_pool
    # Batch-position state is unaffected: the forward batch is still capped at
    # max_batch_size by the micro-batch scheduler.
    assert mgr.get_max_resource_count() == R


@pytest.mark.skipif(not torch.cuda.is_available(), reason="MTP hidden-state pools are CUDA tensors")
def test_mtp_slot_pool_survives_a_full_overlap_turnover():
    """R retiring + R admitted must both hold slots at once.

    This is the exact interleaving the overlap scheduler produces and the one
    that used to exhaust the pool.
    """
    mgr = MTPHiddenStatesManager(
        _mtp_config(), torch.float16, hidden_size=8, max_num_requests=R, num_seq_slots=POOL
    )

    retiring = [mgr.slot_manager.add_slot(rid) for rid in range(R)]
    # Replacements are admitted before the deferred teardown frees the slots.
    incoming = [mgr.slot_manager.add_slot(rid) for rid in range(R, 2 * R)]

    assert len(set(retiring) | set(incoming)) == 2 * R
    assert all(0 <= slot < POOL + 1 for slot in retiring + incoming)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
