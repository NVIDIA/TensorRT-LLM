# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU unit tests for the one-model rejection slot-buffer allocation and the
fail-closed acceptance guard.

These exercise the pure buffer/guard logic with synthetic tensors and lightweight
attribute stand-ins (``types.SimpleNamespace``), so they do not construct a full
``SpecMetadata`` or run any model forward:

- ``SpecMetadata.prepare_rejection_sampling_buffers`` only reads/writes ``self``
  attributes (``use_rejection_sampling``, ``vocab_size``, ``max_num_requests``,
  ``max_draft_len``, ``draft_probs``, ``batch_slot_ids``, ``full_draft_probs``).
- ``SpecWorkerBase._rejection_buffers_valid`` only reads its arguments and
  ``spec_metadata`` attributes (no ``self`` use), so it can be called unbound.

Requires CUDA because the buffers are allocated on ``device='cuda'``.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative.interface import SpecMetadata, SpecWorkerBase

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="rejection buffers are CUDA tensors"
)

R, K, V = 8, 4, 32  # max_num_requests, max_draft_len, vocab_size


def _alloc_meta(**over):
    base = dict(
        use_rejection_sampling=True,
        draft_probs=None,
        vocab_size=V,
        draft_vocab_size=V,
        max_num_requests=R,
        max_draft_len=K,
        batch_slot_ids=None,
        full_draft_probs=None,
        draft_probs_vocab_size=0,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def test_prepare_buffers_allocates_when_enabled():
    # Shared draft/target vocab: draft_probs + batch_slot_ids are allocated;
    # full_draft_probs is skipped (only needed when the vocabs differ).
    m = _alloc_meta()
    SpecMetadata.prepare_rejection_sampling_buffers(m)
    assert m.draft_probs is not None and tuple(m.draft_probs.shape) == (R + 1, K, V)
    assert m.batch_slot_ids is not None and m.batch_slot_ids.shape[0] == R
    assert m.batch_slot_ids.dtype == torch.long
    assert m.full_draft_probs is None
    assert m.draft_probs_vocab_size == V


def test_prepare_buffers_allocates_full_draft_probs_on_vocab_mismatch():
    # Distinct draft vocab: full_draft_probs (d2t-expanded) is allocated.
    m = _alloc_meta(draft_vocab_size=V - 1)
    SpecMetadata.prepare_rejection_sampling_buffers(m)
    assert m.full_draft_probs is not None and tuple(m.full_draft_probs.shape) == (R + 1, K, V)


def test_prepare_buffers_noop_when_disabled():
    m = _alloc_meta(use_rejection_sampling=False)
    SpecMetadata.prepare_rejection_sampling_buffers(m)
    assert m.draft_probs is None
    assert m.batch_slot_ids is None
    assert m.full_draft_probs is None


def _valid_state():
    return types.SimpleNamespace(
        draft_probs=torch.empty((R, K, V), device="cuda"),
        batch_slot_ids=torch.arange(R, device="cuda", dtype=torch.long),
    )


# num_contexts=0, num_gens=4
_NUM_CTX = 0
_BATCH = 4
_NUM_GENS = _BATCH - _NUM_CTX


def _good_args():
    draft_tokens = torch.zeros((_NUM_GENS, K), dtype=torch.int, device="cuda")
    logits = torch.zeros((_NUM_CTX + _NUM_GENS * (K + 1), V), device="cuda")
    return draft_tokens, K, V, _NUM_CTX, _BATCH, logits


_DUMMY = object()  # _rejection_buffers_valid does not use self


def _call(meta, draft_tokens, draft_len, stored_vocab, num_contexts, batch_size, logits):
    return SpecWorkerBase._rejection_buffers_valid(
        _DUMMY, draft_tokens, draft_len, stored_vocab, num_contexts, batch_size, logits, meta
    )


def test_guard_true_on_valid_state():
    assert _call(_valid_state(), *_good_args()) is True


def test_guard_false_when_draft_probs_missing():
    m = _valid_state()
    m.draft_probs = None
    assert _call(m, *_good_args()) is False


def test_guard_false_when_batch_slot_ids_missing():
    m = _valid_state()
    m.batch_slot_ids = None
    assert _call(m, *_good_args()) is False


def test_guard_false_when_stored_vocab_nonpositive():
    dt, dl, _sv, nc, bs, lg = _good_args()
    assert _call(_valid_state(), dt, dl, 0, nc, bs, lg) is False


def test_guard_false_when_stored_vocab_exceeds_buffer():
    dt, dl, _sv, nc, bs, lg = _good_args()
    assert _call(_valid_state(), dt, dl, V + 1, nc, bs, lg) is False


def test_guard_false_when_draft_len_exceeds_buffer():
    # draft_probs has K steps; ask for K+1.
    m = _valid_state()
    draft_tokens = torch.zeros((_NUM_GENS, K + 1), dtype=torch.int, device="cuda")
    logits = torch.zeros((_NUM_CTX + _NUM_GENS * (K + 2), V), device="cuda")
    assert _call(m, draft_tokens, K + 1, V, _NUM_CTX, _BATCH, logits) is False


def test_guard_false_when_draft_tokens_wrong_rows():
    dt = torch.zeros((_NUM_GENS + 1, K), dtype=torch.int, device="cuda")
    _, dl, sv, nc, bs, lg = _good_args()
    assert _call(_valid_state(), dt, dl, sv, nc, bs, lg) is False


def test_guard_false_when_too_few_logits_rows():
    dt, dl, sv, nc, bs, _lg = _good_args()
    too_few = torch.zeros((nc + _NUM_GENS, V), device="cuda")  # missing +1 each
    assert _call(_valid_state(), dt, dl, sv, nc, bs, too_few) is False


def test_guard_false_when_batch_slot_ids_too_short():
    m = _valid_state()
    m.batch_slot_ids = torch.arange(_BATCH - 1, device="cuda", dtype=torch.long)
    assert _call(m, *_good_args()) is False


# --------------------------------------------------------------------------
# Fail-closed acceptance dispatch: prove _accept_draft_tokens() routes to
# strict/base acceptance (and clears draft_probs_valid) when the buffers are
# malformed even though draft_probs_valid was left True, and routes to the
# rejection method when the state is valid. The two acceptance methods are
# stubbed to record which path ran; the real _can_use_rejection_sampling and
# _rejection_buffers_valid are exercised.
# --------------------------------------------------------------------------


class _Worker(SpecWorkerBase):
    @property
    def max_draft_len(self) -> int:
        return K


def _dispatch_meta(**over):
    base = dict(
        use_rejection_sampling=True,
        draft_probs_valid=True,
        is_all_greedy_sample=False,
        draft_probs_vocab_size=V,
        draft_probs_last_dim=V,
        batch_slot_ids=torch.arange(R, device="cuda", dtype=torch.long),
        draft_probs=torch.zeros((R, K, V), device="cuda"),
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _make_worker():
    w = _Worker()
    calls = {"base": 0, "rejection": 0}

    def _base(logits, draft_tokens, num_contexts, batch_size, spec_metadata):
        calls["base"] += 1
        return ("base", None)

    def _rej(logits, draft_tokens, draft_probs, num_contexts, batch_size, spec_metadata):
        calls["rejection"] += 1
        return ("rejection", None)

    w._sample_and_accept_draft_tokens_base = _base
    w._sample_and_accept_draft_tokens_rejection = _rej
    return w, calls


def test_accept_dispatch_fails_closed_on_malformed_buffers():
    w, calls = _make_worker()
    # draft_probs_valid is True, but the buffers are malformed (draft_probs None)
    # -> _rejection_buffers_valid must return False, so acceptance falls back to
    # base, the rejection kernel is skipped, and draft_probs_valid is cleared.
    meta = _dispatch_meta(draft_probs=None)
    draft_tokens = torch.zeros((_NUM_GENS, K), dtype=torch.int, device="cuda")
    logits = torch.zeros((_NUM_CTX + _NUM_GENS * (K + 1), V), device="cuda")
    out = w._accept_draft_tokens(logits, draft_tokens, _NUM_CTX, _BATCH, meta)
    assert out == ("base", None)
    assert calls["base"] == 1 and calls["rejection"] == 0
    assert meta.draft_probs_valid is False  # stale flag cleared on fallback


def test_accept_dispatch_routes_to_rejection_on_valid_state():
    w, calls = _make_worker()
    meta = _dispatch_meta()  # valid buffers
    draft_tokens = torch.zeros((_NUM_GENS, K), dtype=torch.int, device="cuda")
    logits = torch.zeros((_NUM_CTX + _NUM_GENS * (K + 1), V), device="cuda")
    out = w._accept_draft_tokens(logits, draft_tokens, _NUM_CTX, _BATCH, meta)
    assert out == ("rejection", None)
    assert calls["rejection"] == 1 and calls["base"] == 0


def test_accept_dispatch_base_when_all_greedy():
    w, calls = _make_worker()
    # All-greedy batch: rejection is bypassed regardless of buffers.
    meta = _dispatch_meta(is_all_greedy_sample=True)
    draft_tokens = torch.zeros((_NUM_GENS, K), dtype=torch.int, device="cuda")
    logits = torch.zeros((_NUM_CTX + _NUM_GENS * (K + 1), V), device="cuda")
    out = w._accept_draft_tokens(logits, draft_tokens, _NUM_CTX, _BATCH, meta)
    assert out == ("base", None)
    assert calls["base"] == 1 and calls["rejection"] == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
