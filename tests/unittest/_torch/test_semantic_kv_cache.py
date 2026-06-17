# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.semantic_kv_cache import (
    LocalSemanticKvProvider,
    SemanticKvDonor,
    SemanticKvLookupRequest,
    SemanticKvMaterializationKind,
)

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def test_local_semantic_provider_exact_equivalent_result():
    provider = LocalSemanticKvProvider(min_similarity=0.5)
    tokens = [1, 2, 3, 4]
    provider.register_donor(
        SemanticKvDonor(
            donor_id="donor",
            token_ids=tokens,
            prompt_text=None,
            model_id="model",
            namespace="namespace",
        ))

    result = provider.lookup(
        SemanticKvLookupRequest(
            request_id="request",
            token_ids=tokens,
            prompt_text=None,
            model_id="model",
            namespace="namespace",
        ))

    assert result is not None
    assert result.materialization_kind == SemanticKvMaterializationKind.EXACT_PREFIX
    assert result.reusable_token_count == len(tokens)


def test_local_semantic_provider_nonidentical_hit_is_discovery_only():
    provider = LocalSemanticKvProvider(min_similarity=0.5)
    provider.register_donor(
        SemanticKvDonor(
            donor_id="donor",
            token_ids=[1, 2, 3, 4],
            prompt_text=None,
            model_id="model",
            namespace="namespace",
        ))

    result = provider.lookup(
        SemanticKvLookupRequest(
            request_id="request",
            token_ids=[1, 2, 3, 99],
            prompt_text=None,
            model_id="model",
            namespace="namespace",
        ))

    assert result is not None
    assert result.materialization_kind == SemanticKvMaterializationKind.DISCOVERY_ONLY
    assert result.reusable_token_count == 0


def test_local_semantic_provider_is_namespace_isolated():
    provider = LocalSemanticKvProvider(min_similarity=0.1)
    provider.register_donor(
        SemanticKvDonor(
            donor_id="donor",
            token_ids=[1, 2, 3, 4],
            prompt_text=None,
            model_id="model",
            namespace="namespace-a",
        ))

    result = provider.lookup(
        SemanticKvLookupRequest(
            request_id="request",
            token_ids=[1, 2, 3, 4],
            prompt_text=None,
            model_id="model",
            namespace="namespace-b",
        ))

    assert result is None


@dataclass
class _FakeKvCacheConfig:
    tokens_per_block: int


@dataclass
class _FakeLlmArgs:
    kv_cache_config: _FakeKvCacheConfig
    model: str = "fake-model"


class _FakeRequest:

    def __init__(self, request_id: int, tokens: list[int],
                 cache_salt: str | None = None,
                 prompt_len: int | None = None) -> None:
        self.request_id = request_id
        self._tokens = tokens
        self.cache_salt = cache_salt
        if prompt_len is not None:
            self.py_prompt_len = prompt_len

    def get_tokens(self, beam: int) -> list[int]:
        assert beam == 0
        return list(self._tokens)


@dataclass
class _FakeRequestData:
    request_id: int
    new_tokens: list[int]
    new_block_ids: list[int]
    computed_position: int
    num_scheduled_tokens: int
    cache_salt: str | None = None


@dataclass
class _FakeSchedulerOutput:
    new_requests: list[_FakeRequestData]


def _import_semantic_connector_example():
    examples_dir = Path(__file__).parents[3] / "examples" / "llm-api"
    sys.path.insert(0, str(examples_dir))
    try:
        try:
            import llm_semantic_kv_cache_connector as semantic_example
        except ImportError as exc:
            pytest.skip(f"TensorRT-LLM runtime imports unavailable: {exc}")
    finally:
        sys.path.remove(str(examples_dir))
    return semantic_example


def test_semantic_connector_advertises_only_exact_equivalent_loads(
        tmp_path, monkeypatch):
    semantic_example = _import_semantic_connector_example()
    monkeypatch.setenv(semantic_example.SEMANTIC_CONNECTOR_CACHE_FOLDER_KEY,
                       str(tmp_path))
    monkeypatch.setenv(semantic_example.SEMANTIC_CONNECTOR_MODEL_ID_KEY,
                       "fake-model")

    llm_args = _FakeLlmArgs(kv_cache_config=_FakeKvCacheConfig(tokens_per_block=4))
    leader = semantic_example.SemanticPersistentKvCacheConnectorLeader(llm_args)

    donor_tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    donor = _FakeRequest(101, donor_tokens + [90, 91], prompt_len=8)
    leader.request_finished(donor, [0, 1])

    namespace = leader._namespace(None)
    for block_pos in range(2):
        file_path = leader._file_path("101", namespace, block_pos)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"cached-block")

    exact_recipient = _FakeRequest(202, donor_tokens)
    matched, is_async = leader.get_num_new_matched_tokens(exact_recipient, 0)

    assert matched == 8
    assert is_async is False
    assert leader.stats["exact_loads_advertised"] == 1

    metadata = leader.build_connector_meta(
        _FakeSchedulerOutput(new_requests=[
            _FakeRequestData(
                request_id=202,
                new_tokens=donor_tokens,
                new_block_ids=[10, 11],
                computed_position=0,
                num_scheduled_tokens=8,
                cache_salt=None,
            )
        ]))

    assert metadata.load == [
        (str(leader._file_path("101", namespace, 0)), 10),
        (str(leader._file_path("101", namespace, 1)), 11),
    ]

    semantic_recipient = _FakeRequest(303, [1, 2, 3, 4, 5, 6, 7, 99])
    matched, is_async = leader.get_num_new_matched_tokens(semantic_recipient, 0)

    assert matched == 0
    assert is_async is False
    assert leader.stats["discovery_only_hits"] == 1


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_semantic_connector_worker_loads_saved_tensor(tmp_path):
    semantic_example = _import_semantic_connector_example()
    llm_args = _FakeLlmArgs(kv_cache_config=_FakeKvCacheConfig(tokens_per_block=4))
    worker = semantic_example.SemanticPersistentKvCacheConnectorWorker(llm_args)
    kv_cache = torch.zeros(3, 2, 4)
    worker.register_kv_caches(kv_cache)

    source = torch.full((2, 4), 7.0)
    source_path = tmp_path / "block.pt"
    torch.save(source, source_path)

    worker.bind_connector_meta(
        semantic_example.SemanticPersistentKvCacheConnectorMetadata(load=[
            (str(source_path), 1),
        ]))
    worker.start_load_kv(None)

    assert torch.equal(kv_cache[1], source)
