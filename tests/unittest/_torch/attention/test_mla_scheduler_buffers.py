# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The MLA FMHA prefixes must follow post-`prepare()` edits to the lengths.

`mla_cu_q_rows` / `mla_cu_kv_seqlens` replace a per-layer in-kernel scan, so they
are built once and reused across the 60+ MLA layers of an iteration. What makes
that unsafe is that spec-dec and the overlap scheduler rewrite the lengths
*after* `prepare()` and *in place on the device*:

  eagle3.py:1110      `_seq_lens[:batch].fill_(1)`      (q_len 4 -> 1 for drafts)
  eagle3.py:1122      `kv_lens_cuda -= draft - accepted`
  mtp.py:880          `_seq_lens[nc:] -= 1`
  model_engine.py:3388 `kv_lens_cuda += previous_kv_lens_offsets_cuda`

so a host snapshot taken in `prepare()` sees none of it, and FMHA gets request
boundaries that disagree with the packed Q/KV. These tests pin the two properties
that make the reuse safe: the invalidation hooks fire, and the rebuild reads the
device tensors rather than a host mirror.
"""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata

NUM_HEADS = 128


class _Prefixes:
    """The state `mla_prepare_*` touches, with the real methods bound to it.

    Deliberately not a full `TrtllmAttentionMetadata`: that needs a KV cache
    manager and a resource pool, none of which these methods read. Binding the
    production functions keeps this a test of the shipped code rather than of a
    reimplementation.
    """

    # Bind the unbound functions so any edit to them is exercised here.
    mla_prepare_scheduler_buffers = TrtllmAttentionMetadata.mla_prepare_scheduler_buffers
    mla_prepare_ctx_cu_seqlens = TrtllmAttentionMetadata.mla_prepare_ctx_cu_seqlens
    _invalidate_mla_scheduler_buffers = TrtllmAttentionMetadata._invalidate_mla_scheduler_buffers
    on_update_kv_lens = TrtllmAttentionMetadata.on_update_kv_lens
    update_for_spec_dec = TrtllmAttentionMetadata.update_for_spec_dec

    def __init__(self, q_lens, kv_lens, num_contexts=0):
        device = torch.device("cuda")
        self.num_contexts = num_contexts
        self.num_seqs = len(q_lens)
        self.seq_lens_cuda = torch.tensor(q_lens, dtype=torch.int32, device=device)
        self.kv_lens_cuda = torch.tensor(kv_lens, dtype=torch.int32, device=device)
        size = self.num_seqs + 1
        self.mla_cu_q_rows = torch.zeros(size, dtype=torch.int32, device=device)
        self.mla_cu_kv_seqlens = torch.zeros(size, dtype=torch.int32, device=device)
        self.mla_ctx_cu_q_seqlens = torch.zeros(size, dtype=torch.int32, device=device)
        # `enable_flash_mla` is the only other attribute the two hooks read.
        self.enable_flash_mla = False
        self._invalidate_mla_scheduler_buffers()

    def rebuild(self):
        cu_q, cu_kv = self.mla_prepare_scheduler_buffers(NUM_HEADS)
        return cu_q.tolist(), cu_kv.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_q_rows_follow_a_spec_dec_query_length_change():
    """MTP3 verify pass runs q_len 4; the draft sub-steps run 3, then 1.

    Without the invalidation the second call returns the first call's prefixes,
    because the validity flag used to be reset only in `prepare()`.
    """
    state = _Prefixes(q_lens=[4, 4], kv_lens=[100, 200])
    cu_q, _ = state.rebuild()
    assert cu_q == [0, 4 * NUM_HEADS, 8 * NUM_HEADS]

    # What MTPWorker.change_attn_metadata does to the generation rows.
    state.seq_lens_cuda -= 1
    assert state.rebuild()[0] == cu_q, "no rebuild is expected until a hook fires"

    state.update_for_spec_dec()
    assert state.rebuild()[0] == [0, 3 * NUM_HEADS, 6 * NUM_HEADS]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_kv_prefix_follows_an_in_place_device_kv_len_change():
    """The overlap scheduler bumps `kv_lens_cuda` on device, never the host copy.

    A host-derived prefix cannot see this, which is why the rebuild reads
    `kv_lens_cuda` directly.
    """
    state = _Prefixes(q_lens=[1, 1], kv_lens=[100, 200])
    assert state.rebuild()[1] == [0, 100, 300]

    state.kv_lens_cuda += torch.tensor([3, 5], dtype=torch.int32, device="cuda")
    state.on_update_kv_lens()
    assert state.rebuild()[1] == [0, 103, 308]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_partial_acceptance_differs_per_request():
    """Acceptance is per request, so a uniform q_len is not a safe assumption."""
    state = _Prefixes(q_lens=[4, 4, 4], kv_lens=[100, 200, 300])
    state.rebuild()

    state.seq_lens_cuda.copy_(torch.tensor([1, 3, 4], dtype=torch.int32, device="cuda"))
    state.update_for_spec_dec()
    cu_q, _ = state.rebuild()
    assert cu_q == [0, 1 * NUM_HEADS, 4 * NUM_HEADS, 8 * NUM_HEADS]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_context_prefix_counts_tokens_not_rows():
    """`mla_ctx_cu_q_seqlens` feeds the Q RoPE fold, which is indexed by token."""
    state = _Prefixes(q_lens=[5, 3, 7], kv_lens=[5, 3, 7], num_contexts=3)
    assert state.mla_prepare_ctx_cu_seqlens().tolist() == [0, 5, 8, 15]

    state.seq_lens_cuda.copy_(torch.tensor([2, 3, 7], dtype=torch.int32, device="cuda"))
    state.on_update_kv_lens()
    assert state.mla_prepare_ctx_cu_seqlens().tolist() == [0, 2, 5, 12]
