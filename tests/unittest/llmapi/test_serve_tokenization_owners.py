# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Each conversation is tokenized by the frontend that owns it.

Two frontends are simulated in one process: each has its own cache and serves it
on its own socket, as two trtllm-serve frontend processes do.
"""

import os
import socket
import time
from typing import Any

import pytest

from tensorrt_llm.inputs.prefix_token_cache import (
    PrefixTokenCache,
    PrefixTokenCacheConfig,
    tokenization_routing_key,
)
from tensorrt_llm.serve.tokenization_owners import SOCKET_NAME, TokenizationOwners, owner_of

pytestmark = pytest.mark.cpu_only


class _CharTokenizer:
    is_fast = True

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "input_ids": [ord(c) for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


def _cache() -> PrefixTokenCache:
    return PrefixTokenCache(
        PrefixTokenCacheConfig(
            max_entries=64, overlap=4, resync=2, min_chars=64, checkpoint_tokens=8
        )
    )


def _turns(n: int = 6) -> list[str]:
    text, out = "system preamble " * 8, []
    for i in range(n):
        text += f"turn {i} says something new. "
        out.append(text)
    return out


def _key_owned_by(owner: int, num_frontends: int) -> str:
    return next(
        key
        for key in (f"conversation-{i}" for i in range(100))
        if owner_of(key, num_frontends) == owner
    )


def test_a_conversation_is_tokenized_by_its_owner_wherever_its_turns_arrive(tmp_path) -> None:
    tokenizer = _CharTokenizer()
    caches = [_cache(), _cache()]
    owners = [
        TokenizationOwners(str(tmp_path), frontend_id, 2, cache, tokenizer)
        for frontend_id, cache in enumerate(caches)
    ]
    try:
        key = _key_owned_by(1, 2)
        turns = _turns()
        # Turns alternate between the frontends, as SO_REUSEPORT spreads them.
        for i, prompt in enumerate(turns):
            with tokenization_routing_key(key):
                assert caches[i % 2].encode(tokenizer, prompt) == [ord(c) for c in prompt]

        assert caches[0].routed == len(turns) // 2
        assert caches[0].hits + caches[0].misses == 0
        # The owner saw every turn, so every turn after the first hit.
        assert caches[1].hits == len(turns) - 1

        # A prompt without a conversation stays where it arrived.
        caches[0].encode(tokenizer, turns[0])
        assert caches[0].misses == 1
    finally:
        for owner in owners:
            owner.close()


class _FailingTokenizer(_CharTokenizer):
    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        raise ValueError("cannot tokenize")


def test_a_prompt_is_tokenized_locally_when_its_owner_does_not_tokenize_it(tmp_path) -> None:
    tokenizer, cache = _CharTokenizer(), _cache()
    owners = [TokenizationOwners(str(tmp_path), 0, 2, cache, tokenizer, retry_after_s=0.0)]
    try:
        prompt = _turns(1)[0]
        with tokenization_routing_key(_key_owned_by(1, 2)):
            # Frontend 1 has not started.
            assert cache.encode(tokenizer, prompt) == [ord(c) for c in prompt]
            assert (cache.routed, cache.route_failures, cache.misses) == (0, 1, 1)

            # Frontend 1 runs but its tokenizer fails; the connection survives.
            owners.append(TokenizationOwners(str(tmp_path), 1, 2, _cache(), _FailingTokenizer()))
            for _ in range(2):
                assert cache.encode(tokenizer, prompt) == [ord(c) for c in prompt]
        assert (cache.routed, cache.route_failures) == (0, 3)
        assert "3 route failures" in cache.summary()
    finally:
        for owner in owners:
            owner.close()


def test_a_hung_owner_is_skipped_after_it_times_out(tmp_path) -> None:
    tokenizer, cache = _CharTokenizer(), _cache()
    # Frontend 1 accepts connections but never answers.
    hung = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    hung.bind(os.path.join(str(tmp_path), SOCKET_NAME.format(1)))
    hung.listen()
    owners = TokenizationOwners(str(tmp_path), 0, 2, cache, tokenizer, timeout_s=0.5)
    try:
        prompt = _turns(1)[0]
        with tokenization_routing_key(_key_owned_by(1, 2)):
            for _ in range(3):
                start = time.monotonic()
                assert cache.encode(tokenizer, prompt) == [ord(c) for c in prompt]
                elapsed = time.monotonic() - start
            # Only the first prompt waited for the owner.
            assert elapsed < 0.25
        assert (cache.routed, cache.route_failures) == (0, 3)
    finally:
        owners.close()
        hung.close()


def test_a_conversation_id_with_a_lone_surrogate_has_an_owner() -> None:
    # JSON can carry one; it must not fail the request it arrives with.
    assert owner_of("conversation-\ud800", 2) in (0, 1)
