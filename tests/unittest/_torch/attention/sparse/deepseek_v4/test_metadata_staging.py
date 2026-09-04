# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host staging lifetime tests for DSA and DeepSeek-V4 metadata."""

import torch

from tensorrt_llm._torch.attention_backend.sparse.dsa.metadata import (
    _PREPARE_HOST_STAGE_FIELDS,
    DSAtrtllmAttentionMetadata,
)


class _FakeCudaEvent:
    """Host-only event model for the prepare staging ring tests."""

    def __init__(self):
        self.complete = False
        self.query_count = 0
        self.record_count = 0
        self.synchronize_count = 0

    def query(self):
        self.query_count += 1
        return self.complete

    def record(self):
        self.record_count += 1
        self.complete = False

    def synchronize(self):
        self.synchronize_count += 1
        self.complete = True


def _staging_metadata():
    meta = DSAtrtllmAttentionMetadata.__new__(DSAtrtllmAttentionMetadata)
    # One base-class source and one ragged source are enough to prove that a
    # slot switch rebinds the whole audited version together.
    meta.prompt_lens_cpu = torch.zeros(4, dtype=torch.int32)
    meta.host_gen_token_repeats = torch.zeros(4, dtype=torch.int64)
    meta._reset_prepare_host_stage_ring()
    return meta


def _install_fake_cuda_events(monkeypatch):
    events = []

    def event_factory():
        event = _FakeCudaEvent()
        events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", event_factory)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return events


def test_prepare_host_stage_ring_avoids_prior_step_wait(monkeypatch):
    """Two outstanding prepares use distinct host storage without a wait."""
    meta = _staging_metadata()
    events = _install_fake_cuda_events(monkeypatch)
    calls = []

    def prepare_impl(self):
        value = len(calls) + 1
        self.prompt_lens_cpu.fill_(value)
        self.host_gen_token_repeats.fill_(value)
        calls.append((self.prompt_lens_cpu, self.host_gen_token_repeats))

    monkeypatch.setattr(DSAtrtllmAttentionMetadata, "_prepare_impl", prepare_impl)

    meta.prepare()
    first_prompt, first_repeats = calls[0]
    meta.prepare()

    assert calls[1][0] is not first_prompt
    assert calls[1][1] is not first_repeats
    assert first_prompt.tolist() == [1] * 4
    assert first_repeats.tolist() == [1] * 4
    assert len(events) == 2
    assert sum(event.synchronize_count for event in events) == 0


def test_prepare_host_stage_ring_waits_only_when_lapped(monkeypatch):
    """The third prepare fails safe when the first slot is still in flight."""
    meta = _staging_metadata()
    events = _install_fake_cuda_events(monkeypatch)
    slots = []

    def prepare_impl(self):
        slots.append(self.prompt_lens_cpu)

    monkeypatch.setattr(DSAtrtllmAttentionMetadata, "_prepare_impl", prepare_impl)

    meta.prepare()
    meta.prepare()
    meta.prepare()

    assert slots[2] is slots[0]
    assert events[0].query_count == 1
    assert events[0].synchronize_count == 1
    assert events[1].synchronize_count == 0


def test_prepare_host_stage_ring_reuses_completed_slot_without_wait(monkeypatch):
    """A completed slot is recycled after a non-blocking event query."""
    meta = _staging_metadata()
    events = _install_fake_cuda_events(monkeypatch)
    monkeypatch.setattr(DSAtrtllmAttentionMetadata, "_prepare_impl", lambda self: None)

    meta.prepare()
    meta.prepare()
    events[0].complete = True
    meta.prepare()

    assert events[0].query_count == 1
    assert events[0].synchronize_count == 0


def test_prepare_host_stage_ring_drains_before_buffer_replacement(monkeypatch):
    """A rare staging-buffer resize cannot orphan an in-flight source."""
    meta = _staging_metadata()
    events = _install_fake_cuda_events(monkeypatch)
    monkeypatch.setattr(DSAtrtllmAttentionMetadata, "_prepare_impl", lambda self: None)

    meta.prepare()
    replacement = torch.zeros(8, dtype=torch.int32)
    meta.prompt_lens_cpu = replacement
    meta.prepare()

    assert events[0].query_count == 1
    assert events[0].synchronize_count == 1
    assert meta.prompt_lens_cpu is replacement


def test_prepare_host_stage_audit_covers_ragged_and_deepseek_v4_sources():
    """Keep virtual-prepare staging additions inside the versioned set."""
    required = {
        "host_gen_token_repeats",
        "kv_lens_expanded_host",
        "host_block_table_expanded",
        "row_kv_lens_host",
        "row_kv_correction_host",
        "row_req_idx_host",
        "attn_row_kv_lens_host",
        "attn_row_kv_correction_host",
        "attn_row_req_idx_host",
        "attn_row_prompt_lens_cpu",
        "cached_token_lens_cpu",
        "cu_seq_lens",
    }
    assert required <= set(_PREPARE_HOST_STAGE_FIELDS)
    assert len(_PREPARE_HOST_STAGE_FIELDS) == len(set(_PREPARE_HOST_STAGE_FIELDS))
