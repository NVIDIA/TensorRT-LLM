# Copyright (c) 2026, NVIDIA CORPORATION.
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

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.runtime.kv_cache_hash import KV_CACHE_HASH_ALGO_V1, KV_CACHE_HASH_ALGO_V2
from tensorrt_llm.serve.openai_server import OpenAIServer


def _make_server(*, max_batch_size=None, kv_cache_config=None):
    server = object.__new__(OpenAIServer)
    server.generator = SimpleNamespace(
        disaggregated_params=None,
        args=SimpleNamespace(
            max_batch_size=max_batch_size,
            kv_cache_config=kv_cache_config,
        ),
    )
    return server


@pytest.mark.asyncio
async def test_server_info_includes_effective_kv_cache_hash_algo():
    server = _make_server(kv_cache_config=KvCacheConfig())

    response = await server.get_server_info()
    content = json.loads(response.body)
    assert content["kv_cache_hash_algo"] == KV_CACHE_HASH_ALGO_V1

    server.generator.args.kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True)
    response = await server.get_server_info()
    content = json.loads(response.body)
    assert content["kv_cache_hash_algo"] == KV_CACHE_HASH_ALGO_V1

    server.generator.args.kv_cache_config = KvCacheConfig(
        use_kv_cache_manager_v2=True, kv_cache_event_hash_algo=KV_CACHE_HASH_ALGO_V2
    )
    response = await server.get_server_info()
    content = json.loads(response.body)
    assert content["kv_cache_hash_algo"] == KV_CACHE_HASH_ALGO_V2


@pytest.mark.asyncio
async def test_server_info_includes_max_batch_size_for_router_normalization():
    server = _make_server(max_batch_size=256, kv_cache_config=KvCacheConfig())

    response = await server.get_server_info()
    content = json.loads(response.body)
    assert content["max_batch_size"] == 256


@pytest.mark.asyncio
async def test_server_info_omits_max_batch_size_when_none():
    server = _make_server(max_batch_size=None, kv_cache_config=KvCacheConfig())

    response = await server.get_server_info()
    content = json.loads(response.body)
    assert "max_batch_size" not in content


@pytest.mark.asyncio
async def test_server_info_includes_tokens_per_block_from_kv_cache_config():
    server = _make_server(kv_cache_config=KvCacheConfig(tokens_per_block=64))

    response = await server.get_server_info()
    content = json.loads(response.body)
    assert content["tokens_per_block"] == 64
