# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 as resource_manager
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
    gen_multimodal_cache_key_tokens,
)

pytestmark = pytest.mark.cpu_only


_HASH_INTS = (1, 2, 3, 4, 5, 6, 7, 8)
_OTHER_HASH_INTS = (8, 7, 6, 5, 4, 3, 2, 1)


def _make_manager(vocab_size: int):
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.vocab_size = vocab_size
    return manager


def _make_request(tokens, **kwargs):
    return LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=tokens,
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        **kwargs,
    )


def test_gen_multimodal_cache_key_tokens_uses_token_offset():
    vocab_size = 1000
    digest = b"".join(v.to_bytes(4, "big", signed=True) for v in _HASH_INTS)

    assert gen_multimodal_cache_key_tokens(vocab_size, digest, 3, token_offset=2) == [
        vocab_size + 2,
        vocab_size + 3,
        vocab_size + 4,
    ]


def test_augment_tokens_for_block_reuse_uses_exact_multimodal_runs():
    vocab_size = 1000
    digest = b"".join(v.to_bytes(4, "big", signed=True) for v in _HASH_INTS)
    mm_tokens = gen_multimodal_cache_key_tokens(vocab_size, digest, 4)

    tokens = list(range(12))
    manager = _make_manager(vocab_size)
    req = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[4],
        multimodal_item_run_cu_offsets=[0, 2],
        multimodal_run_positions=[1, 8],
        multimodal_run_lengths=[2, 2],
    )

    augmented = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req)
    assert augmented[1:3] == mm_tokens[0:2]
    assert augmented[8:10] == mm_tokens[2:4]
    assert augmented[3:8] == tokens[3:8]

    sliced = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=7, end=10)
    assert sliced == [tokens[7], mm_tokens[2], mm_tokens[3]]


def test_augment_tokens_for_block_reuse_uses_two_item_exact_multimodal_runs():
    vocab_size = 1000
    digest_a = b"".join(v.to_bytes(4, "big", signed=True) for v in _HASH_INTS)
    digest_b = b"".join(v.to_bytes(4, "big", signed=True) for v in _OTHER_HASH_INTS)
    mm_tokens_a = gen_multimodal_cache_key_tokens(vocab_size, digest_a, 3)
    mm_tokens_b = gen_multimodal_cache_key_tokens(vocab_size, digest_b, 3)

    tokens = list(range(16))
    manager = _make_manager(vocab_size)
    req = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS, _OTHER_HASH_INTS],
        multimodal_positions=[1, 9],
        multimodal_lengths=[3, 3],
        multimodal_item_run_cu_offsets=[0, 2, 4],
        multimodal_run_positions=[1, 6, 9, 12],
        multimodal_run_lengths=[2, 1, 1, 2],
    )

    augmented = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req)
    assert augmented[1:3] == mm_tokens_a[0:2]
    assert augmented[6] == mm_tokens_a[2]
    assert augmented[9] == mm_tokens_b[0]
    assert augmented[12:14] == mm_tokens_b[1:3]
    assert augmented[3:6] == tokens[3:6]
    assert augmented[10:12] == tokens[10:12]

    sliced = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=8, end=13)
    assert sliced == [tokens[8], mm_tokens_b[0], tokens[10], tokens[11], mm_tokens_b[1]]


def test_augment_tokens_for_block_reuse_skips_out_of_slice_runs(monkeypatch):
    calls = []

    def fake_gen_multimodal_cache_key_tokens(vocab_size, digest, num_tokens, token_offset=0):
        calls.append((vocab_size, digest, num_tokens, token_offset))
        return [digest, *range(vocab_size + 1, vocab_size + num_tokens)]

    monkeypatch.setattr(
        resource_manager, "gen_multimodal_cache_key_tokens", fake_gen_multimodal_cache_key_tokens
    )

    tokens = list(range(12))
    manager = _make_manager(1000)
    req = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[4],
        multimodal_item_run_cu_offsets=[0, 2],
        multimodal_run_positions=[1, 8],
        multimodal_run_lengths=[2, 2],
    )

    sliced = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=3, end=8)
    assert sliced == tokens[3:8]
    assert calls == []


def test_augment_tokens_for_block_reuse_rejects_incomplete_run_metadata():
    tokens = list(range(8))
    manager = _make_manager(1000)
    req = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[4],
        multimodal_item_run_cu_offsets=[0, 1],
        multimodal_run_positions=None,
        multimodal_run_lengths=None,
    )

    with pytest.raises(ValueError, match="provided together"):
        KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=1, end=3)


def test_block_reuse_metadata_rejects_non_cpu_tensors():
    values = torch.empty(1, dtype=torch.int64, device="meta")
    with pytest.raises(ValueError, match="must be CPU-resident"):
        resource_manager._ensure_int64_cpu_tensor(values)


def test_hash_to_digest_rejects_malformed_hashes():
    with pytest.raises(ValueError, match="Expected 8 int32 hash values"):
        resource_manager._hash_to_digest([1, 2, 3])
    with pytest.raises(ValueError, match="Expected multimodal hash values to be integers"):
        resource_manager._hash_to_digest([*_HASH_INTS[:-1], "8"])


def test_augment_tokens_for_block_reuse_preserves_supplied_item_digest():
    vocab_size = 1000
    tokens = list(range(8))
    manager = _make_manager(vocab_size)

    def augmented_tokens(uuid):
        req = _make_request(
            tokens,
            multimodal_hashes=[_HASH_INTS],
            multimodal_positions=[2],
            multimodal_lengths=[3],
            multimodal_uuids=[uuid],
            multimodal_item_run_cu_offsets=None,
            multimodal_run_positions=None,
            multimodal_run_lengths=None,
        )
        return KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req)

    no_uuid = augmented_tokens(None)
    uuid_a = augmented_tokens("image-a")
    uuid_b = augmented_tokens("image-b")

    content_digest = resource_manager._hash_to_digest(_HASH_INTS)
    assert no_uuid[2:5] == gen_multimodal_cache_key_tokens(vocab_size, content_digest, 3)
    # Input preprocessing owns item identity, including any UUID contribution.
    # The cache must use that digest without applying another UUID hash.
    assert uuid_a == no_uuid == uuid_b


def test_augment_tokens_for_block_reuse_canonicalizes_adjacent_runs():
    tokens = list(range(8))
    manager = _make_manager(1000)
    augmented = []
    for positions, lengths in [([1], [4]), ([1, 3], [2, 2])]:
        req = _make_request(
            tokens,
            multimodal_hashes=[_HASH_INTS],
            multimodal_positions=[1],
            multimodal_lengths=[4],
            multimodal_item_run_cu_offsets=[0, len(positions)],
            multimodal_run_positions=positions,
            multimodal_run_lengths=lengths,
        )
        augmented.append(KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req))

    digest = resource_manager._hash_to_digest(_HASH_INTS)
    assert augmented[0] == augmented[1] == [0, digest, 1001, 1002, 1003, 5, 6, 7]


@pytest.mark.parametrize("hybrid", [False, True])
def test_multimodal_events_keep_chunked_commit_incremental(hybrid):
    tokens = list(range(12))
    req = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[8],
        multimodal_item_run_cu_offsets=None,
        multimodal_run_positions=None,
        multimodal_run_lengths=None,
    )
    manager = _make_manager(1000)
    calls = []
    kv_cache = SimpleNamespace(num_committed_tokens=0, stop_committing=Mock())

    def commit(chunk):
        calls.append(chunk)
        kv_cache.num_committed_tokens += len(chunk)

    kv_cache.commit = commit
    manager.enable_block_reuse = True
    manager.is_draft = False
    manager.event_manager = object()
    manager.kv_cache_map = {req.py_request_id: kv_cache}
    manager._reuse_token_source = lambda request: tokens
    manager._augment_tokens_for_block_reuse = Mock(wraps=manager._augment_tokens_for_block_reuse)
    manager.local_num_mamba_layers = 0
    manager._mark_context_position_as_history = Mock()

    request = SimpleNamespace(
        py_request_id=req.py_request_id,
        is_dummy_request=False,
        multimodal_hashes=req.multimodal_hashes,
        multimodal_positions=req.multimodal_positions,
        multimodal_lengths=req.multimodal_lengths,
        multimodal_item_run_cu_offsets=None,
        multimodal_run_positions=None,
        multimodal_run_lengths=None,
        expect_snapshot_points=[4, 8, 12],
        prompt_len=12,
        get_tokens=lambda beam: tokens,
    )
    commit_blocks = (
        MambaHybridCacheManagerV2.try_commit_blocks
        if hybrid
        else KVCacheManagerV2.try_commit_blocks
    )
    for end in (4, 8, 12):
        request.context_current_position = end
        request.context_remaining_length = len(tokens) - end
        commit_blocks(manager, request)

    digest = resource_manager._hash_to_digest(_HASH_INTS)
    assert calls == [[0, digest, 1001, 1002], [1003, 1004, 1005, 1006], [1007, 9, 10, 11]]
    assert [call.kwargs for call in manager._augment_tokens_for_block_reuse.call_args_list] == [
        {"start": 0, "end": 4},
        {"start": 4, "end": 8},
        {"start": 8, "end": 12},
    ]
    kv_cache.stop_committing.assert_called_once_with()


def test_augment_tokens_for_block_reuse_keeps_contiguous_metadata_path():
    vocab_size = 1000
    digest = b"".join(v.to_bytes(4, "big", signed=True) for v in _HASH_INTS)
    mm_tokens = gen_multimodal_cache_key_tokens(vocab_size, digest, 3)

    tokens = list(range(8))
    manager = _make_manager(vocab_size)
    req = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS],
        multimodal_positions=[2],
        multimodal_lengths=[3],
        multimodal_item_run_cu_offsets=None,
        multimodal_run_positions=None,
        multimodal_run_lengths=None,
    )

    sliced = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=1, end=5)
    assert sliced == [tokens[1], *mm_tokens]
