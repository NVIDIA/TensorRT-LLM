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
"""Centralized, push-based, KV-cache-aware router.

Workers push KV-cache events and load to a shared router over ZMQ; callers query
``select_worker(namespace, block_hashes)`` to place a request on the instance
with the best cache locality and capacity. See ``design`` doc for the full
rationale.
"""

from tensorrt_llm.serve.router_utils import PrefixBlockSet

from .messages import KvCacheEventReport, Selection, WorkerLoadReport
from .reporter import WorkerReporter
from .router_core import (CentralizedKVCacheRouter,
                          CentralizedKVCacheRouterCore, block_key_hasher,
                          score_kv_aware_candidates)
from .zmq_server import KVCacheRouterServer

__all__ = [
    "CentralizedKVCacheRouterCore",
    "CentralizedKVCacheRouter",  # backward-compatible alias of the core
    "KVCacheRouterServer",
    "WorkerReporter",
    "PrefixBlockSet",
    "KvCacheEventReport",
    "WorkerLoadReport",
    "Selection",
    "block_key_hasher",
    "score_kv_aware_candidates",
]
