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
"""Registry of named KV cache connector presets.

Each entry maps a short name to the import path and class names needed
by KvCacheConnectorConfig. The connector module is NOT imported here —
it is resolved at runtime via importlib in py_executor_creator.py.
"""

CONNECTOR_REGISTRY: dict[str, dict[str, str]] = {
    "lmcache": {
        "connector_module": "lmcache.integration.tensorrt_llm.tensorrt_adapter",
        "connector_scheduler_class": "LMCacheKvConnectorScheduler",
        "connector_worker_class": "LMCacheKvConnectorWorker",
    },
    "kvbm": {
        "connector_module": "kvbm.trtllm_integration.connector",
        "connector_scheduler_class": "DynamoKVBMConnectorLeader",
        "connector_worker_class": "DynamoKVBMConnectorWorker",
    },
}
