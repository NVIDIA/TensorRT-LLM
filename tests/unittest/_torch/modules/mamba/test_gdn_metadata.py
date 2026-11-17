# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDNMambaMetadata: the per-iteration state the GDN mixer reads besides the chunked-scan layout."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fla import gdn_metadata
from tensorrt_llm._torch.modules.fla.gdn_metadata import GDNMambaMetadata, gdn_metadata_for_batch
from tensorrt_llm._torch.modules.fla.inforward_snapshot import InforwardSnapshotEntry
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_CONV_WINDOW = 3


class _CacheManager:
    """State slots by request id, and the in-forward snapshot capacity of a GDN manager."""

    inforward_snapshot_rows = 2
    inforward_snapshot_max_tail = 4
    inforward_snapshot_min_main = _CONV_WINDOW
    conv_state_shape = [16, _CONV_WINDOW]

    def __init__(self, slots, plan=None):
        self.slots = slots
        self.inforward_snapshot_plan = plan or {}

    def get_state_indices(self, request_ids, is_padding):
        return [self.slots[rid] for rid in request_ids]

    def state_index(self, request_id):
        return self.slots[request_id]


def _attn_metadata(manager, context_lens, cached, num_decodes):
    request_ids = list(manager.slots)[: len(context_lens) + num_decodes]
    seq_lens = torch.tensor(context_lens + [1] * num_decodes, dtype=torch.int)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=len(context_lens),
        num_ctx_tokens=sum(context_lens),
        kv_cache_manager=manager,
        request_ids=request_ids,
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.tensor(cached + [9] * num_decodes, dtype=torch.int)
        ),
    )


def test_prepare_resets_the_iteration_state_and_aligns_the_decode_slots(monkeypatch):
    invalidated = []
    monkeypatch.setattr(
        gdn_metadata, "invalidate_int32_cu_seqlens_cache", lambda: invalidated.append(True)
    )
    manager = _CacheManager({10: 4, 11: 1, 12: 6, 13: 0, 14: 3})
    metadata = GDNMambaMetadata(max_batch_size=8, chunk_size=64)
    metadata.gdn_iteration_context = object()
    metadata.state_reset_done = True

    metadata.prepare(_attn_metadata(manager, [20, 30], cached=[0, 8], num_decodes=3))
    torch.cuda.synchronize()

    assert invalidated == [True]
    assert metadata.gdn_iteration_context is None
    assert metadata.state_reset_done is False
    # The first context request starts without a state.
    assert metadata.prefill_needs_state_reset is True
    assert metadata.state_indices_decode[:3].tolist() == [6, 0, 3]
    assert metadata.inforward_split is None
    # Without replay slots of its own the manager's replay slots are the state slots.
    assert metadata.replay_state_indices is metadata.state_indices


def test_state_reset_gate_skips_continuation_chunks_except_in_warmup():
    manager = _CacheManager({10: 4, 11: 1, 12: 6})
    metadata = GDNMambaMetadata(max_batch_size=4, chunk_size=64)

    metadata.prepare(_attn_metadata(manager, [20, 30], cached=[5, 8], num_decodes=1))
    assert metadata.prefill_needs_state_reset is False

    with Mamba2Metadata.force_initial_states_for_warmup():
        metadata.prepare(_attn_metadata(manager, [20, 30], cached=[5, 8], num_decodes=1))
    assert metadata.prefill_needs_state_reset is True

    metadata.prepare(_attn_metadata(manager, [], cached=[], num_decodes=3))
    assert metadata.prefill_needs_state_reset is False


def test_prepare_splits_the_planned_requests_for_in_forward_snapshots():
    plan = {11: InforwardSnapshotEntry(row=0, point=26, num_before=26, num_after=4)}
    manager = _CacheManager({10: 4, 11: 1}, plan=plan)
    metadata = GDNMambaMetadata(max_batch_size=4, chunk_size=64)

    metadata.prepare(_attn_metadata(manager, [20, 30], cached=[0, 0], num_decodes=0))
    torch.cuda.synchronize()

    split = metadata.inforward_split
    assert (split.k, split.num_tail_tokens) == (1, 4)
    # Request 11's chunk [20, 50) runs as [20, 46) and the tail [46, 50).
    assert split.cu.tolist() == [0, 20, 46, 50]
    assert split.gather.tolist() == [4, 1, 1]
    assert split.tail_tok.tolist() == [46, 47, 48, 49]
    assert split.conv_win.tolist() == [43, 44, 45]


def test_warmup_split_reaches_prepare_from_any_resolved_class():
    """The warmup resolves the model's metadata class; its in-forward warmup switch reaches
    the GDN metadata's prepare even when that class is a subclass."""

    class ModelMetadata(GDNMambaMetadata):
        pass

    manager = _CacheManager({10: 4, 11: 1, 12: 6})
    metadata = ModelMetadata(max_batch_size=4, chunk_size=64)
    attn = _attn_metadata(manager, [20, 2, 30], cached=[0, 0, 0], num_decodes=0)

    with ModelMetadata.inforward_split_for_warmup():
        metadata.prepare(attn)
    torch.cuda.synchronize()
    split = metadata.inforward_split
    # Both requests with room for the longest tail are split, the 2-token one is not.
    assert split.k == 2
    assert split.stage_rows.tolist() == [0, 1]
    # Requests 10 and 12 are split: their tails are tokens 16-19 and 48-51 of the batch.
    assert split.tail_tok.tolist() == [16, 17, 18, 19, 48, 49, 50, 51]

    metadata.prepare(attn)
    assert metadata.inforward_split is None


class _ModelMetadata(GDNMambaMetadata):
    pass


@pytest.mark.parametrize("metadata_cls", [GDNMambaMetadata, _ModelMetadata])
def test_metadata_rebuilt_for_the_batch_capacity_is_prepared_and_shared(metadata_cls):
    """A rebuilt metadata keeps the GDN class in use and is prepared for the batch, and the
    attention metadata holds the instance the model hands its layers."""
    manager = _CacheManager({10: 4, 11: 1})
    attn = _attn_metadata(manager, [], cached=[], num_decodes=2)
    attn.max_num_requests = 2
    attn.mamba_metadata = metadata_cls(max_batch_size=1, chunk_size=128)

    metadata = gdn_metadata_for_batch(attn)
    torch.cuda.synchronize()

    assert type(metadata) is metadata_cls
    assert metadata.max_batch_size == 2
    assert attn.mamba_metadata is metadata
    # Prepared for this batch: the decode rows carry the requests' state slots.
    assert metadata.state_indices[:2].tolist() == [4, 1]


def test_metadata_without_a_mamba_cache_manager_is_rebuilt_unprepared(monkeypatch):
    """Attention metadata without a Mamba cache manager holds False and has no state to prepare."""
    monkeypatch.setattr(
        GDNMambaMetadata, "prepare", lambda self, attn_metadata: pytest.fail("prepared")
    )
    attn = SimpleNamespace(mamba_metadata=False, max_num_requests=2)

    metadata = gdn_metadata_for_batch(attn)

    assert type(metadata) is GDNMambaMetadata
    assert metadata.max_batch_size == 2
    assert attn.mamba_metadata is metadata


def test_metadata_of_the_batch_capacity_is_kept():
    manager = _CacheManager({10: 4})
    attn = _attn_metadata(manager, [], cached=[], num_decodes=1)
    attn.max_num_requests = 2
    current = GDNMambaMetadata(max_batch_size=2, chunk_size=128)
    attn.mamba_metadata = current

    assert gdn_metadata_for_batch(attn) is current
    attn.mamba_metadata = None
    assert gdn_metadata_for_batch(attn) is None
