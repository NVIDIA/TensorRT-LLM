# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""KV cache connector backed by a Mooncake distributed store.

Offloads KV pages to a shared CPU memory pool so a prefix computed by one engine
can be replayed by another, which regular block reuse cannot do because it never
leaves the instance that computed it.

This is a different component from the Mooncake transfer engine that the C++
cache transceiver uses for disaggregated prefill/decode handoff: that moves KV
point to point between two known peers, while this one publishes pages into a
pool addressed by content. The two compose -- a context server can write pages
here and still hand off over NIXL.

Requires ``KVCacheManagerV2``, which is the manager that can describe its pools
to a connector (``register_kv_cache_layout``), and the Mooncake Python bindings
(``pip install mooncake-transfer-engine``).

Enable it with::

    kv_connector_config = KvCacheConnectorConfig(connector="mooncake-store")

with ``MOONCAKE_CONFIG_PATH`` pointing at a Mooncake JSON config.

By default the KV pools themselves are registered with Mooncake, so the store
reads and writes device memory and no copy is added. That needs the HCA to be
able to pin GPU pages -- GPUDirect RDMA, through ``nvidia_peermem`` or dma-buf.
Where it is missing, registration fails on every pool range and the connector
cannot start; ``"stage_through_host": true`` in the JSON config (or
``TRTLLM_MOONCAKE_STORE_STAGE_THROUGH_HOST=1``) then routes pages through a
pinned host buffer instead, so only host memory is ever registered. The stored
bytes are the same either way, so the two modes can share a pool.
"""

from .config import MooncakeStoreConnectorConfig, StoreRole
from .scheduler import MooncakeStoreConnectorScheduler
from .worker import MooncakeStoreConnectorWorker

__all__ = [
    "MooncakeStoreConnectorConfig",
    "MooncakeStoreConnectorScheduler",
    "MooncakeStoreConnectorWorker",
    "StoreRole",
]
