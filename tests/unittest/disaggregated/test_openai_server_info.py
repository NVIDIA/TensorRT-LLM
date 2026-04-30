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


@pytest.mark.asyncio
async def test_server_info_includes_effective_kv_cache_hash_algo():
    server = object.__new__(OpenAIServer)
    server.generator = SimpleNamespace(
        disaggregated_params=None,
        args=SimpleNamespace(kv_cache_config=KvCacheConfig()),
    )

    response = await server.get_server_info()
    content = json.loads(response.body)
    assert content["kv_cache_hash_algo"] == KV_CACHE_HASH_ALGO_V1

    server.generator.args.kv_cache_config = KvCacheConfig(use_kv_cache_manager_v2=True)
    response = await server.get_server_info()
    content = json.loads(response.body)
    assert content["kv_cache_hash_algo"] == KV_CACHE_HASH_ALGO_V2
