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
_DEFAULT_MM_HASHES = object()


def _hash_ints(base):
    return [base + part_idx for part_idx in range(8)]


def _digest(hash_ints):
    return b"".join(v.to_bytes(4, "big", signed=True) for v in hash_ints)


HASH_INTS = _hash_ints(0x01020300)
DIGEST = _digest(HASH_INTS)
DIGEST_TOKENS = gen_multi_modal_tokens(VOCAB_SIZE, DIGEST, 2)


def _make_request(
    prompt_tokens=None,
    *,
    multimodal_hashes=None,
    multimodal_item_runs=None,
):
    multimodal_prompt_lengths, item_run_spans = _derive_prompt_metadata(multimodal_item_runs)
    sampling_config = tensorrt_llm.bindings.SamplingConfig(SamplingParams()._get_sampling_config())
    return _RequestLike(
        prompt_tokens or PROMPT_TOKENS,
        multimodal_hashes=multimodal_hashes,
        multimodal_item_runs=multimodal_item_runs,
        multimodal_prompt_lengths=multimodal_prompt_lengths,
        multimodal_item_run_spans=item_run_spans,
        sampling_config=sampling_config,
    )


def _augment(req, start: int = 0, end: int | None = None):
    manager = SimpleNamespace(vocab_size=VOCAB_SIZE)
    return KVCacheManagerV2._augment_tokens_for_block_reuse(
        manager, req.get_tokens(0), req, start=start, end=end
    )


def _derive_prompt_metadata(multimodal_item_runs):
    if multimodal_item_runs is None:
        return None, None

    def start_and_length(run):
        if hasattr(run, "prompt_start"):
            return run.prompt_start, run.run_length
        return run[0], run[1]

    multimodal_prompt_lengths = []
    item_run_spans = []
    for item_runs in multimodal_item_runs:
        spans = [start_and_length(run) for run in item_runs]
        multimodal_prompt_lengths.append(sum(length for _, length in spans))
        item_run_spans.append(spans)
    return multimodal_prompt_lengths, item_run_spans


def _make_llm_request(multimodal_item_runs, multimodal_hashes=_DEFAULT_MM_HASHES):
    if multimodal_hashes is _DEFAULT_MM_HASHES:
        multimodal_hashes = [HASH_INTS]
    sampling_config = tensorrt_llm.bindings.SamplingConfig(SamplingParams()._get_sampling_config())
    return LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=PROMPT_TOKENS,
        sampling_config=sampling_config,
        is_streaming=False,
        end_id=2,
        pad_id=0,
        multimodal_hashes=multimodal_hashes,
        multimodal_uuids=None,
        multimodal_item_runs=multimodal_item_runs,
    )


class _RequestLike:
    def __init__(
        self,
        prompt_tokens,
        *,
        multimodal_hashes=None,
        multimodal_item_runs=None,
        multimodal_prompt_lengths=None,
        multimodal_item_run_spans=None,
        sampling_config=None,
    ):
        self._prompt_tokens = prompt_tokens
        self.multimodal_hashes = multimodal_hashes
        self.multimodal_item_runs = multimodal_item_runs
        self.multimodal_prompt_lengths = multimodal_prompt_lengths
        self.multimodal_item_run_spans = multimodal_item_run_spans
        self.sampling_config = sampling_config

    def get_tokens(self, beam_idx):
        return self._prompt_tokens


def test_item_runs_replace_only_exact_sparse_prompt_coverage():
    req = _make_request(
        multimodal_hashes=[HASH_INTS],
        multimodal_item_runs=[[(1, 1, []), (3, 1, [])]],
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


def test_block_reuse_uses_prederived_public_item_run_spans():
    # Intentional mismatch: resource_manager must consume the public
    # LlmRequest-derived span cache, not recompute spans from item runs.
    req = _RequestLike(
        PROMPT_TOKENS,
        multimodal_hashes=[HASH_INTS],
        multimodal_item_runs=[[(1, 2, [])]],
        multimodal_prompt_lengths=[2],
        multimodal_item_run_spans=[[(1, 1), (3, 1)]],
    )

    assert _augment(req) == [
        TEXT_BEFORE,
        DIGEST_TOKENS[0],
        TEXT_GAP,
        DIGEST_TOKENS[1],
        TEXT_AFTER,
    ]


def test_missing_prevalidated_item_run_metadata_is_rejected():
    req = _RequestLike(
        PROMPT_TOKENS,
        multimodal_hashes=[HASH_INTS],
        multimodal_item_runs=[[(1, 1, []), (3, 1, [])]],
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            SamplingParams()._get_sampling_config()
        ),
    )

    with pytest.raises(ValueError, match="prevalidated"):
        _augment(req)


def test_direct_llm_request_allows_item_runs_without_hashes_as_cache_fallback():
    req = _make_llm_request([[(1, 1, [])]], multimodal_hashes=None)

    assert req.multimodal_hashes is None
    assert req.multimodal_embedding_lengths == [1]
    assert req.multimodal_prompt_lengths == [1]
    assert req.multimodal_item_run_spans == [[(1, 1)]]
    assert _augment(req) == PROMPT_TOKENS


def test_same_hash_with_different_exact_runs_has_distinct_cache_keys():
    sparse_req = _make_request(
        multimodal_hashes=[HASH_INTS],
        multimodal_item_runs=[[(1, 1, []), (3, 1, [])]],
    )
    contiguous_req = _make_request(
        multimodal_hashes=[HASH_INTS],
        multimodal_item_runs=[[(1, 2, [])]],
    )

    sparse_tokens = _augment(sparse_req)
    contiguous_tokens = _augment(contiguous_req)

    assert sparse_tokens != contiguous_tokens
    assert _cache_keys(sparse_tokens) != _cache_keys(contiguous_tokens)


def _cache_keys(tokens):
    return [key for _, key in sequence_to_blockchain_keys(2, None, tokens)]
