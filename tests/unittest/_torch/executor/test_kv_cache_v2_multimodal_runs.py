# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

import tensorrt_llm._torch.pyexecutor.resource_manager as resource_manager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
    gen_multimodal_cache_key_tokens,
)

_HASH_INTS = [1, 2, 3, 4, 5, 6, 7, 8]


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

    manager = SimpleNamespace(vocab_size=vocab_size)
    req = SimpleNamespace(
        multimodal_hashes=[_HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[4],
        multimodal_item_run_cu_seqlen=[0, 2],
        multimodal_run_positions=[1, 8],
        multimodal_run_lengths=[2, 2],
    )

    tokens = list(range(12))
    augmented = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req)
    assert augmented[1:3] == mm_tokens[0:2]
    assert augmented[8:10] == mm_tokens[2:4]
    assert augmented[3:8] == tokens[3:8]

    sliced = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=7, end=10)
    assert sliced == [tokens[7], mm_tokens[2], mm_tokens[3]]


def test_augment_tokens_for_block_reuse_skips_out_of_slice_runs(monkeypatch):
    calls = []

    def fake_gen_multimodal_cache_key_tokens(vocab_size, digest, num_tokens, token_offset=0):
        calls.append((vocab_size, digest, num_tokens, token_offset))
        return [digest, *range(vocab_size + 1, vocab_size + num_tokens)]

    monkeypatch.setattr(
        resource_manager, "gen_multimodal_cache_key_tokens", fake_gen_multimodal_cache_key_tokens
    )

    manager = SimpleNamespace(vocab_size=1000)
    req = SimpleNamespace(
        multimodal_hashes=[_HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[4],
        multimodal_item_run_cu_seqlen=[0, 2],
        multimodal_run_positions=[1, 8],
        multimodal_run_lengths=[2, 2],
    )

    tokens = list(range(12))
    sliced = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=3, end=8)
    assert sliced == tokens[3:8]
    assert calls == []


def test_augment_tokens_for_block_reuse_rejects_incomplete_run_metadata():
    manager = SimpleNamespace(vocab_size=1000)
    req = SimpleNamespace(
        multimodal_hashes=[_HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[4],
        multimodal_item_run_cu_seqlen=[0, 1],
    )

    tokens = list(range(8))
    with pytest.raises(ValueError, match="provided together"):
        KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=1, end=3)


def test_augment_tokens_for_block_reuse_keeps_contiguous_metadata_path():
    vocab_size = 1000
    digest = b"".join(v.to_bytes(4, "big", signed=True) for v in _HASH_INTS)
    mm_tokens = gen_multimodal_cache_key_tokens(vocab_size, digest, 3)

    manager = SimpleNamespace(vocab_size=vocab_size)
    req = SimpleNamespace(
        multimodal_hashes=[_HASH_INTS],
        multimodal_positions=[2],
        multimodal_lengths=[3],
    )

    tokens = list(range(8))
    sliced = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=1, end=5)
    assert sliced == [tokens[1], *mm_tokens]
