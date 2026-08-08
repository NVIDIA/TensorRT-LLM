# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compressor's per-request new-token counts must follow the DEVICE buffer.

``_sync_gen_tokens_per_seq`` runs inside the captured ``on_update_kv_lens``,
so its writes are replayed every step. The device-window prologue re-ranks
verify windows from the host shape split S to true windows w by rewriting
``_seq_lens_cuda`` (via ``apply_device_ragged_layout``) after host staging
but before replay. Two properties keep the compressor correct under that
rewrite, and both were violated by the original implementation:

1. The per-request count vector must be a D2D copy sourced from
   ``_seq_lens_cuda`` -- a captured H2D copy from a pinned host list re-loads
   S on every replay, so the kernel computes its state start position
   ``sp = kv_len - nn`` with a w-corrected ``kv_len`` but ``nn = S``, and
   writes the persistent paged KV/score state shifted by (w - S).
2. The ``next_n`` template scalar must be the GLOBAL max window, not this
   step's batch max: the prologue may assign any row up to the config max on
   any bucket, and the kernel treats the scalar as a guarded upper bound.
"""

import types

import torch

from tensorrt_llm._torch.attention_backend.sparse.deepseek_v4.deepseek_v4 import \
    DeepseekV4TrtllmAttentionMetadata


def _dummy(nc, staged_split, seq_lens_dev, max_draft_tokens=5, cap=16):
    d = types.SimpleNamespace()
    d.is_ragged_verify = True
    d.num_generations = len(staged_split)
    d.num_contexts = nc
    d.max_draft_tokens = max_draft_tokens
    # What the host shape split says (S); the fixed code must NOT read it.
    d.ragged_verify_lens = list(staged_split)
    d.gen_new_tokens_per_seq_cuda = torch.zeros(cap, dtype=torch.int)
    d._seq_lens_cuda = torch.tensor(seq_lens_dev, dtype=torch.int)
    return d


def _call(d, num_gen_tokens):
    return DeepseekV4TrtllmAttentionMetadata._sync_gen_tokens_per_seq(
        d, num_gen_tokens)


def test_vector_sources_from_seq_lens_not_host_split():
    # Simulate the prologue: host staged S, device rewrote seq lens to w.
    S = [3, 3, 3]
    w = [2, 4, 3]
    d = _dummy(nc=0, staged_split=S, seq_lens_dev=w)
    _call(d, sum(w))
    assert d.gen_new_tokens_per_seq.tolist() == w


def test_scalar_is_global_bound_not_batch_max():
    S = [2, 2]
    d = _dummy(nc=0, staged_split=S, seq_lens_dev=S)
    next_n = _call(d, sum(S))
    assert next_n == 1 + d.max_draft_tokens
    assert d.num_gen_tokens_per_seq == 1 + d.max_draft_tokens


def test_gen_rows_slice_skips_context_rows():
    ctx_lens = [17, 9]
    w = [4, 2]
    d = _dummy(nc=2, staged_split=w, seq_lens_dev=ctx_lens + w)
    _call(d, sum(w))
    assert d.gen_new_tokens_per_seq.tolist() == w


def test_uniform_branch_unchanged():
    d = _dummy(nc=0, staged_split=[3, 3], seq_lens_dev=[3, 3])
    d.is_ragged_verify = False
    next_n = _call(d, 6)
    assert next_n == 3
    assert d.gen_new_tokens_per_seq is None
