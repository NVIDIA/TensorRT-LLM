# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import gen_multi_modal_tokens

VOCAB_SIZE = 1000
TEXT_BEFORE = 11
MM_PLACEHOLDER = 999
TEXT_GAP = 77
TEXT_AFTER = 12
PROMPT_TOKENS = [
    TEXT_BEFORE,
    MM_PLACEHOLDER,
    TEXT_GAP,
    MM_PLACEHOLDER,
    TEXT_AFTER,
]
HASH_INTS = [
    0x01020304,
    0x05060708,
    0x11121314,
    0x15161718,
    0x21222324,
    0x25262728,
    0x31323334,
    0x35363738,
]
DIGEST = b"".join(v.to_bytes(4, "big", signed=True) for v in HASH_INTS)
DIGEST_TOKENS = gen_multi_modal_tokens(VOCAB_SIZE, DIGEST, 2)


def _augment(req, start: int = 0, end: int | None = None):
    manager = SimpleNamespace(vocab_size=VOCAB_SIZE)
    return KVCacheManagerV2._augment_tokens_for_block_reuse(
        manager, PROMPT_TOKENS, req, start=start, end=end
    )


def test_hash_positions_replace_only_exact_sparse_prompt_positions():
    req = SimpleNamespace(
        multimodal_hashes=[HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[2],
        multimodal_hash_positions=[[1, 3]],
    )

    assert _augment(req) == [
        TEXT_BEFORE,
        DIGEST_TOKENS[0],
        TEXT_GAP,
        DIGEST_TOKENS[1],
        TEXT_AFTER,
    ]
    assert _augment(req, start=2, end=3) == [TEXT_GAP]
    assert _augment(req, start=3, end=4) == [DIGEST_TOKENS[1]]


def test_missing_hash_positions_keeps_legacy_contiguous_projection():
    req = SimpleNamespace(
        multimodal_hashes=[HASH_INTS], multimodal_positions=[1], multimodal_lengths=[2]
    )

    assert _augment(req) == [
        TEXT_BEFORE,
        DIGEST_TOKENS[0],
        DIGEST_TOKENS[1],
        MM_PLACEHOLDER,
        TEXT_AFTER,
    ]
