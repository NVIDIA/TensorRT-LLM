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
'''
Demonstrates using LMCache (https://github.com/LMCache/LMCache) as a KV cache
backend for TensorRT-LLM via the KV Cache Connector interface.

LMCache stores previously computed KV tensors and replays them on subsequent
requests with the same prefix, reducing recomputation.

The connector implementation lives in LMCache:
  lmcache.integration.tensorrt_llm.tensorrt_adapter

TRT-LLM loads it by name at startup via KvCacheConnectorConfig; no code in
this repo needs to change when LMCache is updated.

Prerequisites:
  pip install lmcache

How to run:
  export PYTHONHASHSEED=0
  export LMCACHE_CPU_SIZE_GB=10   # adjust to available RAM (default 20)
  python llm_lmcache_connector.py Qwen/Qwen2-1.5B-Instruct

Expected output:
  Second request logs show "Retrieved N tokens" and both outputs are identical.

See also:
  examples/llm-api/benchmark_trtllm_lmcache_long_doc_qa.sh  -- server benchmark
  examples/llm-api/configs/trtllm_lmcache_connector_extra.yaml -- trtllm-serve YAML
'''

import os

import click

# PYTHONHASHSEED=0 must be set before any hashing occurs so LMCache's
# token-sequence cache keys are stable across calls within the process.
os.environ.setdefault("PYTHONHASHSEED", "0")

from tensorrt_llm import LLM, SamplingParams  # noqa: E402
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, KvCacheConnectorConfig  # noqa: E402

try:
    from lmcache.integration.tensorrt_llm.tensorrt_adapter import (  # noqa: E402
        LMCacheKvConnectorScheduler,
        LMCacheKvConnectorWorker,
        destroy_engine,
    )
except ImportError as e:
    raise ImportError(
        "LMCache is not installed or is missing the TensorRT-LLM integration. "
        "Run: pip install 'lmcache'"
    ) from e

_CONNECTOR_MODULE = "lmcache.integration.tensorrt_llm.tensorrt_adapter"

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
    kv_connector_config = KvCacheConnectorConfig(
        connector_module=_CONNECTOR_MODULE,
        connector_scheduler_class=LMCacheKvConnectorScheduler.__name__,
        connector_worker_class=LMCacheKvConnectorWorker.__name__,
    )
    sampling_params = SamplingParams(max_tokens=32)

    # Both requests go to the same LLM instance so the in-process LMCache
    # engine (and its CPU memory cache) survives between the two calls.
    # TRT-LLM spawns fresh MPI worker processes on every new LLM() call, so
    # in-memory cache state would not survive del llm / new LLM().
    llm = LLM(
        model=model,
        backend="pytorch",
        cuda_graph_config=None,
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
