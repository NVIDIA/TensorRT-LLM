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
"""The one-launch store scatter and the side-stream snapshot hold what four ``index_copy_`` write."""

import types

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler
from tensorrt_llm._torch.speculative import spec_sampler_base as sampler_ops
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler
from tensorrt_llm._torch.speculative.spec_store_scatter import spec_store_scatter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

WIDTH, SLOTS, DRAFT = 6, 8, 5


def _index_copy_store():
    def full(*shape):
        return torch.full(shape, -1, dtype=torch.int32, device="cuda")

    return types.SimpleNamespace(
        new_tokens=full(WIDTH, SLOTS, 1),
        next_new_tokens=full(WIDTH, SLOTS, 1),
        next_draft_tokens=full(SLOTS, DRAFT),
        new_tokens_lens=full(SLOTS),
    )


def test_scatter_matches_index_copy_with_strided_sources():
    gen = torch.Generator(device="cuda").manual_seed(0)

    def ints(*shape):
        return torch.randint(0, 1 << 20, shape, generator=gen, device="cuda", dtype=torch.int32)

    slots = torch.tensor([5, 0, 3], dtype=torch.long, device="cuda")
    new_tokens = ints(4, 2 * WIDTH)[1:, ::2]  # strided rows and columns
    next_new = ints(3, WIDTH)
    lens = ints(6)[::2]
    drafts = ints(3, DRAFT)
    expected = _index_copy_store()
    expected.new_tokens.squeeze(-1).T.index_copy_(0, slots, new_tokens)
    expected.next_new_tokens.squeeze(-1).T.index_copy_(0, slots, next_new)
    expected.new_tokens_lens.index_copy_(0, slots, lens)
    expected.next_draft_tokens.index_copy_(0, slots, drafts)
    store = SpecSampler.Store.allocate(
        num_slots=SLOTS, accepted_len=WIDTH, draft_len=DRAFT, beam_width=1
    )
    store.host_read.fill_(-1)
    store.next_new_tokens.fill_(-1)
    spec_store_scatter(
        slots,
        new_tokens,
        next_new,
        lens,
        drafts,
        store.new_tokens.squeeze(-1),
        store.next_new_tokens.squeeze(-1),
        store.new_tokens_lens,
        store.next_draft_tokens,
    )
    for name in ("new_tokens", "next_new_tokens", "new_tokens_lens", "next_draft_tokens"):
        assert torch.equal(getattr(store, name), getattr(expected, name)), name


def _step(seed, slots=(6, 2)):
    requests = [
        types.SimpleNamespace(
            py_seq_slot=slot,
            py_draft_tokens=[0] * 3,
            py_draft_tokens_effective_len=None,
            py_verify_len=None,
        )
        for slot in slots
    ]
    scheduled = types.SimpleNamespace(
        context_requests_chunking=[], context_requests_last_chunk=[], generation_requests=requests
    )
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def ints(*shape):
        return torch.randint(0, 1000, shape, generator=gen, device="cuda", dtype=torch.int32)

    outputs = {
        "new_tokens": ints(len(slots), 4),
        "new_tokens_lens": ints(len(slots)) % 4 + 1,
        "next_draft_tokens": ints(len(slots), 3),
        "next_new_tokens": ints(len(slots), 4),
        "native_uniform_verify": True,
    }
    return scheduled, outputs


def _expected_host(scheduled, outputs):
    """What the store holds after the step, by four index_copy_ on the old layout."""
    store = _index_copy_store()
    slots = torch.tensor(
        [r.py_seq_slot for r in scheduled.generation_requests], dtype=torch.long, device="cuda"
    )
    pad = torch.nn.functional.pad
    store.new_tokens.squeeze(-1).T.index_copy_(0, slots, pad(outputs["new_tokens"], (0, WIDTH - 4)))
    store.new_tokens_lens.index_copy_(0, slots, outputs["new_tokens_lens"])
    store.next_draft_tokens.index_copy_(0, slots, pad(outputs["next_draft_tokens"], (0, DRAFT - 3)))
    return {name: getattr(store, name).cpu() for name in ("new_tokens", "new_tokens_lens")}, store


def _sampler(monkeypatch):
    monkeypatch.setattr(sampler_ops, "get_draft_token_length", lambda request: 3)
    args = TorchSampler.Args(
        max_seq_len=128,
        max_draft_len=WIDTH - 1,
        max_num_sequences=SLOTS,
        max_beam_width=1,
        max_total_draft_tokens=DRAFT,
    )
    sampler = SpecSampler(args)
    sampler.store.host_read.fill_(-1)
    return sampler


def _assert_step(state, scheduled, outputs):
    want, store = _expected_host(scheduled, outputs)
    slots = [r.py_seq_slot for r in scheduled.generation_requests]
    assert torch.equal(state.host.new_tokens[:, slots], want["new_tokens"][:, slots])
    assert torch.equal(state.host.new_tokens_lens[slots], want["new_tokens_lens"][slots])
    assert torch.equal(state.host.next_draft_tokens[slots], store.next_draft_tokens.cpu()[slots])


def test_sample_async_snapshots_the_store_on_the_side_stream(monkeypatch):
    sampler = _sampler(monkeypatch)
    for seed in (1, 2):  # the second step reuses the batch's slot index
        scheduled, outputs = _step(seed)
        state = sampler.sample_async(scheduled, outputs, [])
        assert state.sampler_event.side_stream_event is not None
        state.sampler_event.synchronize()
        _assert_step(state, scheduled, outputs)
