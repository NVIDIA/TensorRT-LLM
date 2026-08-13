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
### :title LMCache KV Cache Connector
### :order 7
### :section Customization
"""Demonstrates using LMCache as a KV cache backend for TensorRT-LLM.

Uses the KV Cache Connector interface.

LMCache stores previously computed KV tensors and replays them on subsequent
requests with the same prefix, reducing recomputation.

The connector implementation lives in LMCache:
  lmcache.integration.tensorrt_llm.tensorrt_adapter

TRT-LLM resolves the ``"lmcache"`` preset to the correct import paths
automatically via the connector registry.

Prerequisites:
  pip install lmcache

How to run:
  PYTHONHASHSEED=0 python llm_lmcache_connector.py Qwen/Qwen2-1.5B-Instruct

Note: PYTHONHASHSEED=0 must be set before the Python process starts
to ensure deterministic cache key hashing in LMCache.

Expected output:
  Second request logs show "Retrieved N tokens" and both outputs are identical.

See Also:
  examples/llm-api/configs/trtllm_lmcache_connector_extra.yaml -- trtllm-serve YAML
"""

import click

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, KvCacheConnectorConfig

try:
    from lmcache.integration.tensorrt_llm import destroy_engine
except ImportError as e:
    raise ImportError(
        "LMCache is not installed or is missing the TensorRT-LLM integration. "
        "Run: pip install 'lmcache'"
    ) from e

# A prompt long enough to produce at least one full TRT-LLM KV block.
_TEST_PROMPT = (
    "Nvidia Corporation is an American technology company headquartered in "
    "Santa Clara, California. Founded in 1993 by Jensen Huang, Chris "
    "Malachowsky, and Curtis Priem, it develops graphics processing units "
    "(GPUs), system on a chips (SoCs), and application programming "
    "interfaces (APIs) for data science, high-performance computing, and "
    "mobile and automotive applications. Tell me about the company."
)


@click.command()
@click.argument("model", type=str)
def main(model: str):
    kv_cache_config = KvCacheConfig(enable_block_reuse=True)
    kv_connector_config = KvCacheConnectorConfig(connector="lmcache")
    sampling_params = SamplingParams(max_tokens=32)

    # Both requests go to the same LLM instance so the in-process LMCache
    # engine (and its CPU memory cache) survives between the two calls.
    llm = LLM(
        model=model,
        backend="pytorch",
        kv_cache_config=kv_cache_config,
        kv_connector_config=kv_connector_config,
    )

    print("--- First request (cold cache, KV will be computed and stored) ---")
    output0 = llm.generate([_TEST_PROMPT], sampling_params)
    text0 = output0[0].outputs[0].text
    print("First output:", text0)

    print("\n--- Second request (warm cache, KV should be retrieved) ---")
    output1 = llm.generate([_TEST_PROMPT], sampling_params)
    text1 = output1[0].outputs[0].text
    print("Second output (using LMCache KV cache):", text1)

    assert text0 == text1, (
        f"Outputs differ — cache reuse may not have worked correctly.\n"
        f"First:  {text0!r}\n"
        f"Second: {text1!r}"
    )
    print("\nOK: outputs match, LMCache KV reuse confirmed.")

    destroy_engine()


if __name__ == "__main__":
    main()
