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

import contextlib
import os
import socket
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from tensorrt_llm.inputs import prefix_token_cache
from tensorrt_llm.inputs.prefix_token_cache import (
    PrefixTokenCache,
    PrefixTokenCacheConfig,
    tokenization_routing_key,
)
from tensorrt_llm.serve import tokenization_owners
from tensorrt_llm.serve.multi_frontend import (
    MultiFrontendServing,
    frontend_of_conversation,
    tokenization_uds_path,
)
from tensorrt_llm.serve.tokenization_owners import (
    ENABLE_ENV_VAR,
    TokenizationOwners,
    routing_enabled,
    start_tokenization_owners,
)

pytestmark = pytest.mark.cpu_only


class _CharTokenizer:
    is_fast = True

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {
            "input_ids": [ord(c) for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class _RejectingTokenizer(_CharTokenizer):
    """Rejects prompts containing ``MARKER``, as HF tokenizers reject a lone surrogate."""

    MARKER = "<reject>"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        if self.MARKER in text:
            self.calls += 1
            raise TypeError("cannot tokenize")
        return super().__call__(text, **kwargs)


class _SlowTokenizer(_CharTokenizer):
    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s
        self.entered = threading.Event()

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.entered.set()
        time.sleep(self.delay_s)
        return super().__call__(text, **kwargs)


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


def _key_owned_by(owner: int, num_frontends: int, skip: int = 0) -> str:
    keys = (
        key
        for key in (f"conversation-{i}" for i in range(100))
        if frontend_of_conversation(key, num_frontends) == owner
    )
    for _ in range(skip):
        next(keys)
    return next(keys)


@pytest.fixture
def ipc_dir() -> Iterator[str]:
    # Short, so socket paths stay under the AF_UNIX path limit.
    with tempfile.TemporaryDirectory(dir="/tmp") as path:
        yield path


@contextlib.contextmanager
def _owners(ipc_dir: str, *specs: tuple[int, PrefixTokenCache, Any], **kwargs: Any):
    started: list[TokenizationOwners] = []
    try:
        for frontend_id, cache, tokenizer in specs:
            started.append(TokenizationOwners(ipc_dir, frontend_id, 2, cache, tokenizer, **kwargs))
        yield started
    finally:
        for owner in started:
            owner.close()


def test_a_conversation_is_tokenized_by_its_owner_wherever_its_turns_arrive(ipc_dir) -> None:
    tokenizer = _CharTokenizer()
    caches = [_cache(), _cache()]
    with _owners(ipc_dir, (0, caches[0], tokenizer), (1, caches[1], tokenizer)):
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


def test_frontends_share_one_connection_per_owner(ipc_dir) -> None:
    tokenizer = _CharTokenizer()
    caches = [_cache(), _cache()]
    with _owners(ipc_dir, (0, caches[0], tokenizer), (1, caches[1], tokenizer)) as owners:
        prompts = _turns(8)
        errors: list[BaseException] = []

        def worker(thread: int) -> None:
            try:
                with tokenization_routing_key(_key_owned_by(1, 2, skip=thread)):
                    for prompt in prompts:
                        assert caches[0].encode(tokenizer, prompt) == [ord(c) for c in prompt]
            except BaseException as exc:  # surface in the main thread
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert caches[0].routed == 4 * len(prompts)
        assert len(owners[1]._server.handlers) == 1


def test_a_prompt_is_tokenized_locally_while_its_owner_is_not_serving(ipc_dir) -> None:
    tokenizer, cache = _CharTokenizer(), _cache()
    prompt = _turns(1)[0]
    with _owners(ipc_dir, (0, cache, tokenizer)) as owners:
        with tokenization_routing_key(_key_owned_by(1, 2)):
            # Frontend 1 has not started.
            assert cache.encode(tokenizer, prompt) == [ord(c) for c in prompt]
            assert (cache.routed, cache.route_failures, cache.misses) == (0, 1, 1)

            # Once it serves, it is asked at once: a refused connect is not a timeout.
            owners.append(TokenizationOwners(ipc_dir, 1, 2, _cache(), tokenizer))
            assert cache.encode(tokenizer, prompt) == [ord(c) for c in prompt]
        assert (cache.routed, cache.route_failures) == (1, 1)
        assert "1 route failures" in cache.summary()


def test_a_prompt_the_owner_cannot_tokenize_disables_no_cache(ipc_dir) -> None:
    tokenizer, owner_tokenizer = _RejectingTokenizer(), _RejectingTokenizer()
    cache, owner_cache = _cache(), _cache()
    with _owners(ipc_dir, (0, cache, tokenizer), (1, owner_cache, owner_tokenizer)) as owners:
        turns = _turns(3)
        rejected = turns[0] + _RejectingTokenizer.MARKER
        with tokenization_routing_key(_key_owned_by(1, 2)):
            assert cache.encode(tokenizer, turns[0]) == [ord(c) for c in turns[0]]
            connection = owners[0]._peers[1].conn
            calls = owner_tokenizer.calls

            with pytest.raises(TypeError):
                cache.encode(tokenizer, rejected)
            # The owner tried it, with and without the cache, and so did the
            # asking frontend.
            assert owner_tokenizer.calls == calls + 2
            assert tokenizer.calls == 2
            assert not cache.disabled and not owner_cache.disabled
            assert cache.route_failures == 1

            # The connection survives, and the conversation still hits at its owner.
            for prompt in turns[1:]:
                assert cache.encode(tokenizer, prompt) == [ord(c) for c in prompt]
            assert owners[0]._peers[1].conn is connection
        assert cache.routed == 3
        assert owner_cache.hits == 2


def test_a_hung_owner_is_skipped_after_it_times_out(ipc_dir) -> None:
    tokenizer, cache = _CharTokenizer(), _cache()
    timeout_s = 0.5
    # Frontend 1 accepts connections but never answers.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as hung:
        hung.bind(tokenization_uds_path(ipc_dir, 1))
        hung.listen()
        with (
            _owners(ipc_dir, (0, cache, tokenizer), timeout_s=timeout_s),
            patch.object(tokenization_owners.logger, "warning") as warning,
        ):
            prompt = _turns(1)[0]
            with tokenization_routing_key(_key_owned_by(1, 2)):
                elapsed = []
                for _ in range(3):
                    start = time.monotonic()
                    assert cache.encode(tokenizer, prompt) == [ord(c) for c in prompt]
                    elapsed.append(time.monotonic() - start)
            # Only the first prompt waited for the owner.
            assert elapsed[0] >= timeout_s
            assert max(elapsed[1:]) < timeout_s / 2
            assert (cache.routed, cache.route_failures) == (0, 3)
            # Logged once, when the owner started being skipped.
            assert warning.call_count == 1
            assert "Frontend 1" in warning.call_args.args[0]


def test_close_does_not_wait_long_for_a_handler_in_the_middle_of_a_prompt(ipc_dir) -> None:
    tokenizer, slow = _CharTokenizer(), _SlowTokenizer(delay_s=3.0)
    cache = _cache()
    owner = TokenizationOwners(ipc_dir, 1, 2, _cache(), slow)
    with _owners(ipc_dir, (0, cache, tokenizer)):
        prompt = _turns(1)[0]

        def ask() -> None:
            with tokenization_routing_key(_key_owned_by(1, 2)):
                cache.encode(tokenizer, prompt)

        asking = threading.Thread(target=ask, daemon=True)
        asking.start()
        assert slow.entered.wait(5.0)
        start = time.monotonic()
        owner.close(timeout_s=0.2)
        # Up to the serve_forever poll interval (0.5 s) plus the shared deadline.
        assert time.monotonic() - start < 1.5


def test_routed_prompts_count_toward_the_periodic_log_line(ipc_dir) -> None:
    tokenizer = _CharTokenizer()
    caches = [_cache(), _cache()]
    with (
        _owners(ipc_dir, (0, caches[0], tokenizer), (1, caches[1], tokenizer)),
        patch.object(prefix_token_cache, "LOG_EVERY_REQUESTS", 2),
        patch.object(prefix_token_cache.logger, "info") as info,
    ):
        with tokenization_routing_key(_key_owned_by(1, 2)):
            for prompt in _turns(4):
                caches[0].encode(tokenizer, prompt)
    assert caches[0].requests == 4
    # Two lines from frontend 0, whose prompts were all routed, and two from the owner.
    lines = [call.args[0] for call in info.call_args_list]
    assert sum("4 routed" in line or "2 routed" in line for line in lines) == 2
    assert len(lines) == 4


def test_close_removes_the_route(ipc_dir) -> None:
    tokenizer, cache = _CharTokenizer(), _cache()
    with _owners(ipc_dir, (0, cache, tokenizer)):
        pass
    prompt = _turns(1)[0]
    with tokenization_routing_key(_key_owned_by(1, 2)):
        assert cache.encode(tokenizer, prompt) == [ord(c) for c in prompt]
    assert (cache.routed, cache.route_failures, cache.misses) == (0, 0, 1)


def test_the_owner_of_a_conversation_is_fixed() -> None:
    # Disaggregated servers pick the affinity port with the same mapping, in
    # other processes, so it must not change (for example to the seeded hash()).
    assert [frontend_of_conversation(f"conversation-{i}", 8) for i in range(4)] == [2, 4, 6, 0]
    assert frontend_of_conversation("session-a", 3) == 1


def test_a_conversation_id_with_a_lone_surrogate_has_an_owner() -> None:
    # JSON can carry one; it must not fail the request it arrives with.
    assert frontend_of_conversation("conversation-\ud800", 2) in (0, 1)


@pytest.mark.parametrize("value, enabled", [(None, False), ("0", False), ("1", True)])
def test_routing_switch_values(monkeypatch: pytest.MonkeyPatch, value: Any, enabled: bool) -> None:
    if value is None:
        monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENABLE_ENV_VAR, value)
    assert routing_enabled() is enabled


@pytest.mark.parametrize("value", ["true", "on", "1 ", "", "2"])
def test_a_malformed_routing_switch_raises(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(ENABLE_ENV_VAR, value)
    with pytest.raises(ValueError, match=ENABLE_ENV_VAR):
        routing_enabled()


def _processor(cache: Any) -> SimpleNamespace:
    tokenizer = _CharTokenizer()
    return SimpleNamespace(
        get_prefix_token_cache=lambda: None if cache is None else (cache, tokenizer)
    )


def _serving(ipc_dir: str, num_frontends: int, frontend_id: int = 0) -> MultiFrontendServing:
    return MultiFrontendServing(
        launcher_uds=os.path.join(ipc_dir, "launcher.sock"),
        is_launcher=frontend_id == 0,
        frontend_id=frontend_id,
        num_frontends=num_frontends,
    )


def test_routing_starts_when_switched_on_with_frontends_and_a_cache(
    monkeypatch: pytest.MonkeyPatch, ipc_dir: str
) -> None:
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    cache = _cache()
    with patch.object(tokenization_owners.logger, "info") as info:
        owners = start_tokenization_owners(_serving(ipc_dir, 4, frontend_id=2), _processor(cache))
    try:
        assert owners is not None
        assert os.path.exists(tokenization_uds_path(ipc_dir, 2))
        assert "frontend 2 of 4" in info.call_args.args[0]
    finally:
        owners.close()


@pytest.mark.parametrize("num_frontends, has_cache", [(1, True), (4, False)])
def test_routing_warns_when_switched_on_but_it_cannot_engage(
    monkeypatch: pytest.MonkeyPatch, ipc_dir: str, num_frontends: int, has_cache: bool
) -> None:
    monkeypatch.setenv(ENABLE_ENV_VAR, "1")
    processor = _processor(_cache() if has_cache else None)
    with patch.object(tokenization_owners.logger, "warning") as warning:
        assert start_tokenization_owners(_serving(ipc_dir, num_frontends), processor) is None
        assert start_tokenization_owners(None, processor) is None
    assert warning.call_count == 2
    assert ENABLE_ENV_VAR in warning.call_args.args[0]
    assert not os.listdir(ipc_dir)


def test_routing_is_off_by_default(monkeypatch: pytest.MonkeyPatch, ipc_dir: str) -> None:
    monkeypatch.delenv(ENABLE_ENV_VAR, raising=False)
    assert start_tokenization_owners(_serving(ipc_dir, 4), _processor(_cache())) is None
    assert not os.listdir(ipc_dir)
