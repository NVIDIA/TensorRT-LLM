# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate-draft-vocab d2t expansion at the shared rejection acceptance helper.

Methods with a separate (smaller) draft vocabulary (PARD/DFLASH/DRAFT_TARGET)
store draft-vocab probabilities and must expand them to the target vocabulary
only at acceptance: ``_sample_and_accept_draft_tokens_rejection`` scatters the
gathered draft-vocab rows into ``full_draft_probs`` at the d2t-projected target
indices ``(arange(draft_vocab) + d2t) % vocab`` and leaves all other positions
zero. No separate-draft-vocab PARD/DFLASH checkpoint is available in the test
mount, so this drives the REAL shared helper with synthetic tensors and captures
the ``draft_probs`` argument handed to the rejection kernel.
"""

import pytest
import torch

import tensorrt_llm._torch.speculative.interface as iface
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="d2t expansion runs on CUDA tensors"
)


class _Worker(SpecWorkerBase):
    @property
    def max_draft_len(self) -> int:
        return 4


class _Meta:
    """Minimal spec_metadata stand-in for the rejection acceptance helper."""

    def __init__(self, full_draft_probs, d2t):
        self.full_draft_probs = full_draft_probs
        self.d2t = d2t
        self.d2t_target_indices = None
        self.skip_top_k = True
        self.skip_top_p = True
        self.top_ks = None
        self.top_ps = None
        self.temperatures = None  # set by the test to match logits rows


def test_block_d2t_expansion_into_full_draft_probs(monkeypatch):
    dev = "cuda"
    num_gens, K, draft_vocab, vocab = 2, 3, 8, 16
    Rmax, Kmax = 4, 4  # oversized full_draft_probs to exercise the runtime slice

    # d2t[i] = i  ->  target index (i + i) % vocab = 2*i, all distinct and < vocab.
    d2t = torch.arange(draft_vocab, dtype=torch.long, device=dev)
    expected_target_indices = (torch.arange(draft_vocab, device=dev) + d2t) % vocab

    # Draft-vocab proposal rows (normalized), already gathered for the gen subset.
    draft_probs = torch.rand(num_gens, K, draft_vocab, device=dev)
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)

    # Pre-zeroed oversized buffer, mirroring prepare()'s one-time zero-fill.
    full_buf = torch.zeros(Rmax, Kmax, vocab, dtype=torch.float32, device=dev)
    meta = _Meta(full_buf, d2t)

    num_gen_logits = num_gens * (K + 1)
    logits = torch.randn(num_gen_logits, vocab, device=dev)
    meta.temperatures = torch.ones(num_gen_logits, device=dev)
    draft_tokens = torch.randint(0, vocab, (num_gens, K), dtype=torch.int32, device=dev)

    captured = {}

    def _capture_kernel(*, draft_probs, draft_token_ids, target_probs, deterministic, seed, offset):
        captured["draft_probs"] = draft_probs.clone()
        accepted = torch.zeros((num_gens, K + 1), dtype=torch.int, device=dev)
        num_acc = torch.ones(num_gens, dtype=torch.int, device=dev)
        return accepted, num_acc

    monkeypatch.setattr(iface, "rejection_sampling_one_model", _capture_kernel)

    worker = _Worker()
    worker._sample_and_accept_draft_tokens_rejection(
        logits, draft_tokens, draft_probs, 0, num_gens, meta
    )

    # d2t target indices computed and cached correctly.
    assert meta.d2t_target_indices is not None
    assert torch.equal(meta.d2t_target_indices.to(dev), expected_target_indices)

    fdp = captured["draft_probs"]  # = full_draft_probs[:num_gens, :K]
    assert tuple(fdp.shape) == (num_gens, K, vocab)
    # Expansion lands the draft-vocab rows at the d2t-projected target indices.
    assert torch.allclose(fdp[:, :, expected_target_indices], draft_probs, atol=1e-6)
    # Every other target-vocab position stays zero (mapped-only writes; no stale).
    mask = torch.ones(vocab, dtype=torch.bool, device=dev)
    mask[expected_target_indices] = False
    assert torch.count_nonzero(fdp[:, :, mask]) == 0


def test_block_d2t_expansion_runtime_slice_excludes_extra_rows(monkeypatch):
    """A runtime_draft_len shorter than the buffer must not leak extra rows."""
    dev = "cuda"
    num_gens, K, draft_vocab, vocab = 1, 2, 4, 8
    Rmax, Kmax = 4, 4

    d2t = torch.arange(draft_vocab, dtype=torch.long, device=dev)
    target_idx = (torch.arange(draft_vocab, device=dev) + d2t) % vocab

    draft_probs = torch.rand(num_gens, K, draft_vocab, device=dev)
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)

    # Poison the rows beyond runtime_draft_len with garbage to ensure the
    # [:, :runtime_draft_len] slice handed to the kernel excludes them.
    full_buf = torch.zeros(Rmax, Kmax, vocab, dtype=torch.float32, device=dev)
    full_buf[:, K:, :] = 7.0
    meta = _Meta(full_buf, d2t)

    num_gen_logits = num_gens * (K + 1)
    logits = torch.randn(num_gen_logits, vocab, device=dev)
    meta.temperatures = torch.ones(num_gen_logits, device=dev)
    draft_tokens = torch.randint(0, vocab, (num_gens, K), dtype=torch.int32, device=dev)

    captured = {}

    def _capture_kernel(*, draft_probs, draft_token_ids, target_probs, deterministic, seed, offset):
        captured["draft_probs"] = draft_probs.clone()
        return (
            torch.zeros((num_gens, K + 1), dtype=torch.int, device=dev),
            torch.ones(num_gens, dtype=torch.int, device=dev),
        )

    monkeypatch.setattr(iface, "rejection_sampling_one_model", _capture_kernel)

    worker = _Worker()
    worker._sample_and_accept_draft_tokens_rejection(
        logits, draft_tokens, draft_probs, 0, num_gens, meta
    )

    fdp = captured["draft_probs"]
    # Exactly runtime_draft_len rows reached the kernel (no poisoned extra rows).
    assert tuple(fdp.shape) == (num_gens, K, vocab)
    assert torch.allclose(fdp[:, :, target_idx], draft_probs, atol=1e-6)
    assert (fdp == 7.0).sum() == 0
