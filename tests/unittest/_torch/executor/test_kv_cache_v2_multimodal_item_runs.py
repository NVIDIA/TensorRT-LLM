# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

import tensorrt_llm
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManagerV2
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
    gen_multi_modal_tokens,
    sequence_to_blockchain_keys,
)
from tensorrt_llm.sampling_params import SamplingParams

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
OTHER_HASH_INTS = [
    0x0A0B0C0D,
    0x0E0F1011,
    0x1A1B1C1D,
    0x1E1F2021,
    0x2A2B2C2D,
    0x2E2F3031,
    0x3A3B3C3D,
    0x3E3F4041,
]


def _digest(hash_ints):
    return b"".join(v.to_bytes(4, "big", signed=True) for v in hash_ints)


DIGEST = _digest(HASH_INTS)
DIGEST_TOKENS = gen_multi_modal_tokens(VOCAB_SIZE, DIGEST, 2)
OTHER_DIGEST = _digest(OTHER_HASH_INTS)
OTHER_DIGEST_TOKENS = gen_multi_modal_tokens(VOCAB_SIZE, OTHER_DIGEST, 3)


def _make_request(
    prompt_tokens=None,
    *,
    multimodal_hashes=None,
    multimodal_positions=None,
    multimodal_lengths=None,
    multimodal_item_runs=None,
):
    return LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=prompt_tokens or PROMPT_TOKENS,
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            SamplingParams()._get_sampling_config()
        ),
        is_streaming=False,
        multimodal_hashes=multimodal_hashes,
        multimodal_positions=multimodal_positions,
        multimodal_lengths=multimodal_lengths,
        multimodal_item_runs=multimodal_item_runs,
    )


def _augment(req, start: int = 0, end: int | None = None):
    manager = SimpleNamespace(vocab_size=VOCAB_SIZE)
    return KVCacheManagerV2._augment_tokens_for_block_reuse(
        manager, req.get_tokens(0), req, start=start, end=end
    )


def test_item_runs_replace_only_exact_sparse_prompt_coverage():
    req = _make_request(
        multimodal_hashes=[HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[2],
        multimodal_item_runs=[[(1, 1), (3, 1)]],
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


def test_missing_item_runs_keeps_legacy_contiguous_projection():
    req = _make_request(
        multimodal_hashes=[HASH_INTS], multimodal_positions=[1], multimodal_lengths=[2]
    )

    assert _augment(req) == [
        TEXT_BEFORE,
        DIGEST_TOKENS[0],
        DIGEST_TOKENS[1],
        MM_PLACEHOLDER,
        TEXT_AFTER,
    ]


def test_multiple_items_replace_sparse_runs_across_slices():
    prompt_tokens = [101, 999, 201, 999, 301, 998, 401, 998, 402, 998, 501]
    req = _make_request(
        prompt_tokens,
        multimodal_hashes=[HASH_INTS, OTHER_HASH_INTS],
        multimodal_positions=[1, 5],
        multimodal_lengths=[2, 3],
        multimodal_item_runs=[[(1, 1), (3, 1)], [(5, 1), (7, 1), (9, 1)]],
    )

    assert _augment(req) == [
        101,
        DIGEST_TOKENS[0],
        201,
        DIGEST_TOKENS[1],
        301,
        OTHER_DIGEST_TOKENS[0],
        401,
        OTHER_DIGEST_TOKENS[1],
        402,
        OTHER_DIGEST_TOKENS[2],
        501,
    ]
    assert _augment(req, start=2, end=6) == [
        201,
        DIGEST_TOKENS[1],
        301,
        OTHER_DIGEST_TOKENS[0],
    ]
    assert _augment(req, start=6, end=10) == [
        401,
        OTHER_DIGEST_TOKENS[1],
        402,
        OTHER_DIGEST_TOKENS[2],
    ]
    assert _augment(req, start=8, end=9) == [402]


@pytest.mark.parametrize(
    "multimodal_item_runs, match",
    [
        ([[(1, 1), (99, 1)]], "outside prompt"),
        ([[(-1, 1), (3, 1)]], "outside prompt"),
        ([[(1, 2), (2, 1)]], "ordered and non-overlapping"),
        ([[(1, 0), (3, 1)]], "positive length"),
        ([[(1, 1)]], "length"),
        ([[(2, 2)]], "start"),
    ],
)
def test_invalid_exact_item_runs_are_rejected(multimodal_item_runs, match):
    req = _make_request(
        multimodal_hashes=[HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[2],
        multimodal_item_runs=multimodal_item_runs,
    )

    with pytest.raises(ValueError, match=match):
        _augment(req)


def test_same_contiguous_fallback_span_with_different_exact_runs_has_distinct_cache_keys():
    sparse_req = _make_request(
        multimodal_hashes=[HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[2],
        multimodal_item_runs=[[(1, 1), (3, 1)]],
    )
    contiguous_req = _make_request(
        multimodal_hashes=[HASH_INTS],
        multimodal_positions=[1],
        multimodal_lengths=[2],
        multimodal_item_runs=[[(1, 2)]],
    )

    sparse_tokens = _augment(sparse_req)
    contiguous_tokens = _augment(contiguous_req)

    assert sparse_tokens != contiguous_tokens
    assert _cache_keys(sparse_tokens) != _cache_keys(contiguous_tokens)


def _cache_keys(tokens):
    return [key for _, key in sequence_to_blockchain_keys(2, None, tokens)]
