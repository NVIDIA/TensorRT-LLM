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
"""The fused SpecSampler store scatter, its single D2H snapshot, the pinned host ring and the kernel-written
host snapshot hold the same values as the index_copy_ + per-tensor D2H path."""

import types

import pytest
import torch

from tensorrt_llm._torch.speculative import spec_sampler_base as sampler_ops
from tensorrt_llm._torch.speculative.spec_sampler_base import SpecSampler
from tensorrt_llm._torch.speculative.spec_store_scatter import spec_store_scatter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

WIDTH, SLOTS, DRAFT = 6, 8, 5


def _stores(flat):
    if not flat:
        return None, types.SimpleNamespace(
            new_tokens=torch.full((WIDTH, SLOTS, 1), -1, dtype=torch.int32, device="cuda"),
            next_new_tokens=torch.full((WIDTH, SLOTS, 1), -1, dtype=torch.int32, device="cuda"),
            next_draft_tokens=torch.full((SLOTS, DRAFT), -1, dtype=torch.int32, device="cuda"),
            new_tokens_lens=torch.full((SLOTS,), -1, dtype=torch.int32, device="cuda"),
        )
    buffer = torch.full(
        (WIDTH * SLOTS + SLOTS + SLOTS * DRAFT,), -1, dtype=torch.int32, device="cuda"
    )
    return buffer, types.SimpleNamespace(
        new_tokens=buffer[: WIDTH * SLOTS].view(WIDTH, SLOTS, 1),
        next_new_tokens=torch.full((WIDTH, SLOTS, 1), -1, dtype=torch.int32, device="cuda"),
        next_draft_tokens=buffer[WIDTH * SLOTS + SLOTS :].view(SLOTS, DRAFT),
        new_tokens_lens=buffer[WIDTH * SLOTS : WIDTH * SLOTS + SLOTS],
    )


def test_scatter_matches_index_copy_with_strided_sources():
    gen = torch.Generator(device="cuda").manual_seed(0)
    slots = torch.tensor([5, 0, 3], dtype=torch.long, device="cuda")
    wide = torch.randint(
        0, 1 << 20, (4, 2 * WIDTH), generator=gen, device="cuda", dtype=torch.int32
    )
    new_tokens = wide[1:, ::2]  # strided rows and columns
    next_new = torch.randint(
        0, 1 << 20, (3, WIDTH), generator=gen, device="cuda", dtype=torch.int32
    )
    lens = torch.randint(1, WIDTH, (6,), generator=gen, device="cuda", dtype=torch.int32)[::2]
    drafts = torch.randint(0, 1 << 20, (3, DRAFT), generator=gen, device="cuda", dtype=torch.int32)
    _, expected = _stores(False)
    expected.new_tokens.squeeze(-1).T.index_copy_(0, slots, new_tokens)
    expected.next_new_tokens.squeeze(-1).T.index_copy_(0, slots, next_new)
    expected.new_tokens_lens.index_copy_(0, slots, lens)
    expected.next_draft_tokens.index_copy_(0, slots, drafts)
    for flat in (False, True):
        _, store = _stores(flat)
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
            assert torch.equal(getattr(store, name), getattr(expected, name)), (flat, name)


def _step(device):
    requests = [
        types.SimpleNamespace(
            py_seq_slot=slot, py_draft_tokens=[0] * 3, py_draft_tokens_effective_len=None
        )
        for slot in (6, 2)
    ]
    scheduled = types.SimpleNamespace(
        context_requests_chunking=[], context_requests_last_chunk=[], generation_requests=requests
    )
    gen = torch.Generator(device=device).manual_seed(1)

    def ints(*shape):
        return torch.randint(0, 1000, shape, generator=gen, device=device, dtype=torch.int32)

    outputs = {
        "new_tokens": ints(2, 4),
        "new_tokens_lens": torch.tensor([2, 4], dtype=torch.int32, device=device),
        "next_draft_tokens": ints(2, 3),
        "next_new_tokens": ints(2, 4),
    }
    return scheduled, outputs


def _bare_sampler(fused):
    sampler = SpecSampler.__new__(SpecSampler)
    sampler.draft_len = DRAFT
    flat, sampler.store = _stores(fused)
    sampler._store_flat = flat
    sampler._store_views = (WIDTH * SLOTS, WIDTH * SLOTS + SLOTS)
    sampler._device_slots = {}
    # As __init__ builds them for the fused store.
    sampler._host_ring = (
        [torch.empty_like(flat, device="cpu", pin_memory=True) for _ in range(3)] if fused else []
    )
    sampler._host_ring_pos = 0
    sampler._store_scatter = None
    sampler._store_copies_done = None
    sampler._async_worker_init(False)
    sampler._record_sampler_event = lambda side_stream_event=None: None
    return sampler


def _no_zero_copy(monkeypatch):
    """Take the fused store's D2H snapshot copy instead of the kernel-written host snapshot."""
    monkeypatch.setattr(sampler_ops, "prefer_pinned", lambda: False)


@pytest.mark.parametrize("fused", [False, True], ids=["index_copy", "fused"])
def test_sample_async_store_and_host_snapshot(monkeypatch, fused):
    """Same store contents and host snapshot through the fused path, in one D2H copy (into the host ring)."""
    monkeypatch.setattr(sampler_ops, "get_draft_token_length", lambda request: 3)
    _no_zero_copy(monkeypatch)
    sampler = _bare_sampler(fused)
    copies = []

    def copy_to_host(tensor):
        copies.append(tuple(tensor.shape))
        return tensor.cpu()

    sampler._copy_to_host = copy_to_host
    state = sampler.sample_async(*_step("cuda"), [])
    reference = _bare_sampler(False)
    reference._copy_to_host = lambda tensor: tensor.cpu()
    expected = reference.sample_async(*_step("cuda"), [])
    torch.cuda.synchronize()
    if fused:
        assert copies == []
    for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
        assert torch.equal(getattr(state.host, name), getattr(expected.host, name)), name
        assert torch.equal(getattr(state.device, name), getattr(expected.device, name)), name
    if fused:
        # The slot index tensor of an unchanged batch is reused.
        sampler.sample_async(*_step("cuda"), [])
        assert len(sampler._device_slots) == 1


def test_host_ring_snapshots(monkeypatch):
    """The fused snapshot rotates through three pinned buffers allocated once, with the same host values as the
    per-step copy."""
    monkeypatch.setattr(sampler_ops, "get_draft_token_length", lambda request: 3)
    _no_zero_copy(monkeypatch)
    sampler = _bare_sampler(True)
    reference = _bare_sampler(False)
    reference._copy_to_host = lambda tensor: tensor.cpu()
    seen = []
    for _ in range(4):
        state = sampler.sample_async(*_step("cuda"), [])
        expected = reference.sample_async(*_step("cuda"), [])
        torch.cuda.synchronize()
        for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
            assert torch.equal(getattr(state.host, name), getattr(expected.host, name)), name
        seen.append(state.host.new_tokens_lens.untyped_storage().data_ptr())
        assert state.host.new_tokens_lens.is_pinned()
    assert seen[0] == seen[3] and len(set(seen[:3])) == 3


def test_fused_store_views_keep_store_dtypes_and_shapes(monkeypatch):
    """The flat-buffer views have exactly the dtypes and shapes of the separate store tensors."""
    from tensorrt_llm._torch.pyexecutor.sampler.sampler import TorchSampler

    args = TorchSampler.Args(
        max_seq_len=128,
        max_draft_len=DRAFT,
        max_num_sequences=SLOTS,
        max_beam_width=1,
        max_total_draft_tokens=DRAFT,
    )
    monkeypatch.setattr(sampler_ops, "_uses_fused_store", lambda: True)
    sampler = SpecSampler(args)
    assert sampler._store_flat is not None
    _, separate_stores = _stores(False)
    for name in ("new_tokens", "next_new_tokens", "new_tokens_lens", "next_draft_tokens"):
        separate, view = getattr(separate_stores, name), getattr(sampler.store, name)
        assert (view.dtype, view.shape, view.device) == (
            separate.dtype,
            separate.shape,
            separate.device,
        ), name


def test_zero_copy_snapshot(monkeypatch):
    """The scatter kernel writes the sampled slots into the ring's pinned buffer itself (no D2H snapshot copy);
    those slots hold the values of the copied snapshot and the device stores are unchanged."""
    monkeypatch.setattr(sampler_ops, "get_draft_token_length", lambda request: 3)
    monkeypatch.setattr(sampler_ops, "prefer_pinned", lambda: True)
    sampler = _bare_sampler(True)

    def no_copy():
        raise AssertionError("D2H snapshot copy used")

    sampler._host_snapshot = no_copy
    reference = _bare_sampler(False)
    reference._copy_to_host = lambda tensor: tensor.cpu()
    slots = [6, 2]  # _step's requests
    seen = []
    for _ in range(4):
        state = sampler.sample_async(*_step("cuda"), [])
        expected = reference.sample_async(*_step("cuda"), [])
        torch.cuda.synchronize()
        assert torch.equal(state.host.new_tokens[:, slots], expected.host.new_tokens[:, slots])
        assert torch.equal(state.host.new_tokens_lens[slots], expected.host.new_tokens_lens[slots])
        assert torch.equal(
            state.host.next_draft_tokens[slots], expected.host.next_draft_tokens[slots]
        )
        for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
            assert torch.equal(getattr(state.device, name), getattr(expected.device, name)), name
        assert state.host.new_tokens_lens.is_pinned()
        seen.append(state.host.new_tokens_lens.untyped_storage().data_ptr())
    assert seen[0] == seen[3] and len(set(seen[:3])) == 3


def _draft_tail_step(device, transposed):
    """A step whose next_new_tokens / next_draft_tokens are the draft loop's token-buffer layout: views
    tok.t() and tok[1:].t() of one [k + 1, b] int32 buffer, i.e. strides (1, b); runtime draft len = store width."""
    requests = [
        types.SimpleNamespace(
            py_seq_slot=slot, py_draft_tokens=[0] * DRAFT, py_draft_tokens_effective_len=None
        )
        for slot in (6, 2, 4)
    ]
    scheduled = types.SimpleNamespace(
        context_requests_chunking=[], context_requests_last_chunk=[], generation_requests=requests
    )
    gen = torch.Generator(device=device).manual_seed(2)
    tok = torch.randint(
        0, 1 << 20, (WIDTH, len(requests)), generator=gen, device=device, dtype=torch.int32
    )
    next_new, next_draft = tok.t(), tok[1:].t()
    if not transposed:
        next_new, next_draft = next_new.contiguous(), next_draft.contiguous()
    else:
        assert next_new.stride() == (1, len(requests)) and next_draft.stride() == (1, len(requests))
    outputs = {
        "new_tokens": torch.randint(
            0, 1 << 20, (len(requests), WIDTH), generator=gen, device=device, dtype=torch.int32
        ),
        "new_tokens_lens": torch.tensor([2, 6, 4], dtype=torch.int32, device=device),
        "next_draft_tokens": next_draft,
        "next_new_tokens": next_new,
    }
    return scheduled, outputs


@pytest.mark.parametrize("mode", ["index_copy", "fused", "fused_zero_copy"])
def test_draft_tail_transposed_outputs(monkeypatch, mode):
    """The draft tail's transposed int32 views feed the store exactly like contiguous outputs, on the index_copy_,
    fused-scatter and zero-copy paths (device stores and host snapshot)."""
    monkeypatch.setattr(sampler_ops, "get_draft_token_length", lambda request: DRAFT)
    monkeypatch.setattr(sampler_ops, "prefer_pinned", lambda: mode == "fused_zero_copy")
    sampler = _bare_sampler(mode != "index_copy")
    sampler._copy_to_host = lambda tensor: tensor.cpu()
    reference = _bare_sampler(False)
    reference._copy_to_host = lambda tensor: tensor.cpu()
    state = sampler.sample_async(*_draft_tail_step("cuda", transposed=True), [])
    expected = reference.sample_async(*_draft_tail_step("cuda", transposed=False), [])
    torch.cuda.synchronize()
    slots = [6, 2, 4]
    for name in ("new_tokens", "new_tokens_lens", "next_draft_tokens"):
        assert torch.equal(getattr(state.device, name), getattr(expected.device, name)), name
    for name in ("new_tokens", "next_new_tokens", "new_tokens_lens", "next_draft_tokens"):
        assert torch.equal(getattr(sampler.store, name), getattr(reference.store, name)), name
    assert torch.equal(state.host.new_tokens[:, slots], expected.host.new_tokens[:, slots])
    assert torch.equal(state.host.new_tokens_lens[slots], expected.host.new_tokens_lens[slots])
    assert torch.equal(state.host.next_draft_tokens[slots], expected.host.next_draft_tokens[slots])
