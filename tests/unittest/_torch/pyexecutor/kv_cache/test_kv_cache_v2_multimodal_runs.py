# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

import tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 as resource_manager
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.sharing._manager import keyed_by_placeholders
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_NO_CONTENT_HASHES_KEY,
    MultimodalParams,
    request_cache_salt,
)
from tensorrt_llm.inputs.registry import create_input_processor_with_hash
from tensorrt_llm.runtime.kv_cache_hash import hash_v1_block_key
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    KVCacheEventManager,
    ReuseScope,
    gen_multimodal_cache_key_tokens,
    sequence_to_blockchain_keys,
)

pytestmark = pytest.mark.cpu_only


_HASH_INTS = (1, 2, 3, 4, 5, 6, 7, 8)
_OTHER_HASH_INTS = (8, 7, 6, 5, 4, 3, 2, 1)


def _make_manager(vocab_size: int):
    manager = KVCacheManagerV2.__new__(KVCacheManagerV2)
    manager.vocab_size = vocab_size
    manager.event_manager = None
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


def test_multimodal_uuid_metadata_does_not_change_digest_token_identity_or_block_keys():
    vocab_size = 1000
    digest = b"".join(v.to_bytes(4, "big", signed=True) for v in _HASH_INTS)
    uuid = "frontend-routing-identity"
    without_uuid = gen_multimodal_cache_key_tokens(vocab_size, digest, 3)
    with_uuid = gen_multimodal_cache_key_tokens(vocab_size, digest, 3, uuid=uuid)
    with_other_uuid = gen_multimodal_cache_key_tokens(
        vocab_size, digest, 3, uuid="other-routing-identity"
    )

    assert with_uuid == with_other_uuid == without_uuid
    assert with_uuid[0].digest == digest
    assert with_uuid[0].uuid == uuid
    assert hash(with_uuid[0]) == hash(with_other_uuid[0]) == hash(without_uuid[0])
    assert len({with_uuid[0], with_other_uuid[0], without_uuid[0]}) == 1
    assert [key for _, key in sequence_to_blockchain_keys(2, ReuseScope(), with_uuid)] == [
        key for _, key in sequence_to_blockchain_keys(2, ReuseScope(), without_uuid)
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


def test_augment_tokens_for_block_reuse_uses_two_item_exact_multimodal_runs(monkeypatch):
    """Keep each item's UUID and token offsets across separated and sliced runs."""
    vocab_size = 1000
    digest_a = b"".join(v.to_bytes(4, "big", signed=True) for v in _HASH_INTS)
    digest_b = b"".join(v.to_bytes(4, "big", signed=True) for v in _OTHER_HASH_INTS)
    mm_tokens_a = gen_multimodal_cache_key_tokens(vocab_size, digest_a, 3)
    mm_tokens_b = gen_multimodal_cache_key_tokens(vocab_size, digest_b, 3)
    generator = Mock(wraps=gen_multimodal_cache_key_tokens)
    monkeypatch.setattr(resource_manager, "gen_multimodal_cache_key_tokens", generator)

    tokens = list(range(16))
    manager = _make_manager(vocab_size)
    req = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS, _OTHER_HASH_INTS],
        multimodal_uuids=["item-a", "item-b"],
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
    assert augmented[1].uuid == "item-a"
    assert augmented[9].uuid == "item-b"
    assert generator.call_args_list == [
        call(vocab_size, digest_a, 2, token_offset=0, uuid="item-a"),
        call(vocab_size, digest_a, 1, token_offset=2, uuid="item-a"),
        call(vocab_size, digest_b, 1, token_offset=0, uuid="item-b"),
        call(vocab_size, digest_b, 2, token_offset=1, uuid="item-b"),
    ]

    generator.reset_mock()
    sliced = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req, start=8, end=13)
    assert sliced == [tokens[8], mm_tokens_b[0], tokens[10], tokens[11], mm_tokens_b[1]]
    assert sliced[1].uuid == "item-b"
    assert generator.call_args_list == [
        call(vocab_size, digest_b, 1, token_offset=0, uuid="item-b"),
        call(vocab_size, digest_b, 1, token_offset=1, uuid="item-b"),
    ]

    generator.reset_mock()
    continuation = KVCacheManagerV2._augment_tokens_for_block_reuse(
        manager, tokens, req, start=2, end=7
    )
    assert continuation == [mm_tokens_a[1], *tokens[3:6], mm_tokens_a[2]]
    assert generator.call_args_list == [
        call(vocab_size, digest_a, 1, token_offset=1, uuid="item-a"),
        call(vocab_size, digest_a, 1, token_offset=2, uuid="item-a"),
    ]


def test_augment_tokens_for_block_reuse_skips_out_of_slice_runs(monkeypatch):
    calls = []

    def fake_gen_multimodal_cache_key_tokens(
        vocab_size, digest, num_tokens, token_offset=0, uuid=None
    ):
        calls.append((vocab_size, digest, num_tokens, token_offset, uuid))
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


def test_augment_tokens_for_block_reuse_preserves_supplied_item_digest_and_uuid():
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
    # The cache must use that digest without applying another UUID hash, while
    # retaining the external identity independently of token equality.
    assert uuid_a == no_uuid == uuid_b
    assert uuid_a[2].digest == content_digest
    assert uuid_a[2].uuid == "image-a"
    assert uuid_b[2].digest == content_digest
    assert uuid_b[2].uuid == "image-b"


def test_augment_tokens_for_block_reuse_handles_partial_uuid_list():
    vocab_size = 1000
    tokens = list(range(10))
    manager = _make_manager(vocab_size)
    req = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS, _OTHER_HASH_INTS],
        multimodal_positions=[1, 6],
        multimodal_lengths=[2, 2],
        multimodal_uuids=["first-item"],
        multimodal_item_run_cu_offsets=None,
        multimodal_run_positions=None,
        multimodal_run_lengths=None,
    )

    augmented = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, req)
    assert augmented[1].uuid == "first-item"
    assert isinstance(augmented[6], bytes)


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
    """Preserve UUID context while committing each prefill chunk only once."""
    tokens = list(range(12))
    req = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS],
        multimodal_uuids=["chunked-item"],
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
    manager._can_publish_block_reuse = True
    manager.is_draft = False
    manager.event_manager = object()
    manager.kv_cache_map = {req.py_request_id: kv_cache}
    manager._reuse_token_source = lambda request: tokens
    manager._augment_tokens_for_block_reuse = Mock(wraps=manager._augment_tokens_for_block_reuse)
    manager.local_num_mamba_layers = 0
    manager._mark_context_position_as_history = Mock()
    # Prompt-only commit: the cache stops at the last context chunk.
    manager._can_commit_generation_blocks = lambda request: False

    request = SimpleNamespace(
        py_request_id=req.py_request_id,
        is_dummy_request=False,
        multimodal_hashes=req.multimodal_hashes,
        multimodal_uuids=req.multimodal_uuids,
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
    assert calls[0][1].uuid == "chunked-item"
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


@pytest.mark.parametrize("exact_runs", [False, True])
def test_native_uuid_context_survives_text_only_first_prefill_chunk(exact_runs):
    tokens = list(range(8))
    request = _make_request(
        tokens,
        multimodal_hashes=[_HASH_INTS],
        multimodal_uuids=["later-image"],
        multimodal_positions=[2],
        multimodal_lengths=[4],
        multimodal_item_run_cu_offsets=[0, 1] if exact_runs else None,
        multimodal_run_positions=[2] if exact_runs else None,
        multimodal_run_lengths=[4] if exact_runs else None,
    )
    manager = _make_manager(1000)
    chunks = [
        manager._augment_tokens_for_block_reuse(tokens, request, start=start, end=start + 2)
        for start in range(0, len(tokens), 2)
    ]
    assert chunks[0] == tokens[:2]
    assert chunks[1][0].uuid == "later-image"
    assert chunks[2] == [1002, 1003]
    assert sum(chunks, []) == manager._augment_tokens_for_block_reuse(tokens, request)


class _ImageProcessor:
    """Four multimodal tokens (101-104) after one text token.

    Counts four tokens per image, or two per image and two per video for a mixed request. Like
    Phi-4-MM, it has no audio token count.
    """

    def __init__(
        self, hashing_supported: bool | None, count_error: Exception | None = None
    ) -> None:
        self.multimodal_hashing_supported = hashing_supported
        self.count_error = count_error

    def __call__(self, inputs: dict, sampling_params: object) -> tuple[list[int], dict]:
        return [10, 101, 102, 103, 104, 20, 21, 22], {"multimodal_data": {}}

    def get_num_tokens_per_image(self, *, image: np.ndarray) -> int:
        if self.count_error is not None:
            raise self.count_error
        return 4

    def get_num_tokens_per_video(self, *, video: list[np.ndarray], **kwargs: object) -> int:
        return 2

    def get_vocab_size(self) -> int:
        return 100

    def get_mm_token_ids(self) -> None:
        return None

    def get_mm_special_token_ids(self) -> None:
        return None


def _image(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)


# Each case: the processor, then two requests with other media behind the same prompt tokens.
_NO_CONTENT_HASH_CASES = {
    "hashing_off": (
        lambda: _ImageProcessor(hashing_supported=False),
        [{"image": [_image(value)]} for value in (0, 255)],
    ),
    "token_count_error": (
        lambda: _ImageProcessor(True, count_error=ValueError("other max_pixels")),
        [{"image": [_image(value)]} for value in (0, 255)],
    ),
    "audio_without_token_count": (
        lambda: _ImageProcessor(hashing_supported=True),
        [{"audio": [np.full(16, value, dtype=np.float32)]} for value in (0.0, 1.0)],
    ),
    "mixed_without_item_order": (
        lambda: _ImageProcessor(hashing_supported=True),
        [{"image": [_image(value)], "video": [[_image(value)]]} for value in (0, 255)],
    ),
}


@pytest.mark.parametrize("case", list(_NO_CONTENT_HASH_CASES))
def test_requests_without_content_hashes_do_not_share_block_keys(case: str) -> None:
    """Other media, same prompt tokens, no content hashes: no block key is shared.

    Keyed by placeholder tokens alone, the second request would reuse the KV cache of the first
    request's media. Items whose prompt spans are found get random hashes; a request whose spans
    are not found (the token count fails, audio without a token count, mixed modalities without
    an item order) gets a cache salt of its own. Checked for the V2 keys and, where the keys are
    text only, for the V1 keys.
    """
    make_processor, requests = _NO_CONTENT_HASH_CASES[case]
    wrapped = create_input_processor_with_hash(make_processor())
    manager = _make_manager(100)
    v2_keys, v1_keys = [], []
    for mm_data in requests:
        tokens, extra = wrapped({"prompt": "unused", "multi_modal_data": mm_data}, None)
        mm_input = extra.get("multimodal_input")
        mm_fields = {}
        if mm_input is not None:
            mm_fields = dict(
                multimodal_hashes=mm_input.multimodal_hashes,
                multimodal_positions=mm_input.multimodal_positions,
                multimodal_lengths=mm_input.multimodal_lengths,
                **mm_input.run_metadata(),
            )
        # The salt LLM.generate_async submits the request with.
        params = MultimodalParams(
            multimodal_input=mm_input, multimodal_data=extra["multimodal_data"]
        )
        request = _make_request(tokens, cache_salt=request_cache_salt(None, params), **mm_fields)
        scope = ReuseScope(
            request.lora_task_id, KVCacheManagerV2._derive_reuse_salt(request.cache_salt)
        )
        augmented = KVCacheManagerV2._augment_tokens_for_block_reuse(manager, tokens, request)
        keys = sequence_to_blockchain_keys(4, scope, augmented)
        v2_keys.append([key for block, key in keys if block])  # Skip the root key.
        if mm_input is None:
            salt = request.cache_salt
            salt_id = None if salt is None else KVCacheEventManager._hash_cache_salt(salt)
            v1_keys.append([hash_v1_block_key(tokens[:4], cache_salt_id=salt_id)])
            v1_keys[-1].append(hash_v1_block_key(tokens[4:], v1_keys[-1][0]))

    assert [len(keys) for keys in v2_keys] == [2, 2]
    assert not set(v2_keys[0]) & set(v2_keys[1])
    if case != "hashing_off":
        assert len(v1_keys) == 2
        assert not set(v1_keys[0]) & set(v1_keys[1])


@pytest.mark.parametrize(
    "py_multimodal_data,expected",
    [
        ({"image": {}}, False),
        ({"image": {}, MULTIMODAL_NO_CONTENT_HASHES_KEY: True}, True),
    ],
    ids=["content_hashes", "random_hashes"],
)
def test_kv_lender_treats_random_hashes_like_no_hashes(py_multimodal_data, expected) -> None:
    """The KV lender does not lend blocks whose multimodal keys do not name the content."""
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    request = SimpleNamespace(
        py_multimodal_data=py_multimodal_data,
        multimodal_hashes=[list(_HASH_INTS)],
        multimodal_positions=[1],
        multimodal_lengths=[4],
    )

    assert keyed_by_placeholders(request) is expected
    PyExecutor._release_multimodal_resources(None, request)
    assert "image" not in request.py_multimodal_data
    assert keyed_by_placeholders(request) is expected
