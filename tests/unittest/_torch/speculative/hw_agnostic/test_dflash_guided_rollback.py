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
"""GPU unit tests: DFlash speculation under guided decoding.

``CapturableGuidedDecoder.execute`` advances every grammar matcher through the
golden position and all draft tokens (as far as the grammar allows) to mask the
target logits. Drafters with a per-step drafted-logits loop undo the rejected
tail in ``execute_draft_batch(draft_step=0)``; DFlash drafts a whole block from
hidden states and never enters that loop, so ``DFlashWorker`` has to roll the
matchers back itself through ``rollback_rejected_batch``, exactly as
``SAWorker`` does (see ``test_sa_guided_rollback.py``, which also pins the
decoder-level rollback semantics with a real xgrammar matcher). Without it the
next verification masks from a grammar state that still contains rejected
draft tokens: grammar-forbidden tokens get through, and once a rejected draft
token terminated the matcher, masking stops entirely.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.llmapi import DFlashDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the DFlash worker runs on CUDA tensors",
)

VOCAB_PADDED = 64
K = 3  # draft tokens per step


class _RecordingGuidedDecoder:
    def __init__(self):
        self.calls = []

    def execute(self, logits, d2t=None):
        self.calls.append(("execute", None))

    def rollback_rejected_batch(self, num_accepted_tokens):
        self.calls.append(("rollback_rejected_batch", num_accepted_tokens.clone()))


def _worker_inputs(batch_size, num_contexts, runtime_draft_len):
    meta = types.SimpleNamespace(
        runtime_draft_len=runtime_draft_len,
        is_cuda_graph=False,
        request_ids=None,
        batch_indices_cuda=None,
    )
    attn = types.SimpleNamespace(
        num_seqs=batch_size,
        num_contexts=num_contexts,
        kv_cache_manager=None,
        has_spec_dec_saved_state=False,
    )
    input_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    position_ids = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    hidden = torch.zeros(batch_size, 8, device="cuda")
    logits = torch.zeros(batch_size, VOCAB_PADDED, device="cuda")
    return meta, attn, input_ids, position_ids, hidden, logits


def _noop(*args, **kwargs):
    return None


def test_dflash_worker_rolls_back_after_verification(monkeypatch):
    worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=K), Mapping())
    guided = _RecordingGuidedDecoder()
    worker.guided_decoder = guided

    # A context-only batch skips the draft forward but still runs the guided
    # mask and the verification, so the call order is pinned without loading a
    # draft model. The buffer bookkeeping around them is stubbed out.
    batch_size = num_contexts = 2
    meta, attn, *inputs = _worker_inputs(batch_size, num_contexts, runtime_draft_len=K)
    worker._ctx_len = torch.zeros(batch_size, dtype=torch.long, device="cuda")
    # _lazy_init_ctx_buffers (stubbed here) normally builds the host mirror.
    worker._ctx_len_host = [0] * batch_size
    accepted = torch.zeros(batch_size, K + 1, dtype=torch.int32, device="cuda")
    # Distinct per-row counts: one row accepts the whole draft, the other only
    # the golden token. Uniform values would let a broken forward that drops or
    # transposes rows still pass the equality below.
    num_accepted = torch.tensor([K + 1, 1], dtype=torch.int32, device="cuda")
    next_new = torch.zeros(batch_size, K + 1, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(
        worker, "sample_and_accept_draft_tokens", lambda *a, **k: (accepted, num_accepted)
    )
    for stub in (
        "_lazy_init_ctx_buffers",
        "_prepare_attn_metadata_for_dflash",
        "_prepare_kv_for_draft_forward",
        "_store_prefill_context",
        "write_context_onehot_draft_probs",
        "_restore_attn_metadata_from_spec_dec",
    ):
        monkeypatch.setattr(worker, stub, _noop)

    def finish_drafting(*args, **kwargs):
        guided.calls.append(("draft_complete", None))

    monkeypatch.setattr(worker, "_apply_kv_rewind_after_draft", finish_drafting)
    monkeypatch.setattr(worker, "prepare_1st_drafter_inputs", lambda *a, **k: {})
    monkeypatch.setattr(worker, "_prepare_next_new_tokens", lambda *a, **k: next_new)

    out = worker.forward(*inputs, attn, meta, draft_model=None)

    # Target mask first, then exactly one rollback fed the verification's own counts.
    assert [name for name, _ in guided.calls] == [
        "execute",
        "draft_complete",
        "rollback_rejected_batch",
    ]
    assert torch.equal(guided.calls[2][1], num_accepted)
    assert torch.equal(out["new_tokens_lens"], num_accepted)


def test_dflash_worker_skips_the_rollback_when_not_drafting(monkeypatch):
    """A zero runtime draft length verifies nothing, so there is nothing to roll back."""
    worker = DFlashWorker(DFlashDecodingConfig(max_draft_len=K), Mapping())
    guided = _RecordingGuidedDecoder()
    worker.guided_decoder = guided

    batch_size = 2
    meta, attn, *inputs = _worker_inputs(batch_size, 0, runtime_draft_len=0)
    sampled = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(worker, "_sample_tokens_for_batch", lambda *a, **k: sampled)

    out = worker.forward(*inputs, attn, meta, draft_model=None)

    assert [name for name, _ in guided.calls] == ["execute"]
    assert out["next_draft_tokens"].shape == (batch_size, 0)
