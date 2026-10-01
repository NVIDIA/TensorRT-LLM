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

Offloads KV pages to a shared CPU memory pool addressed by content, so a prefix
computed by one engine can be replayed by another. Distinct from the Mooncake
transfer engine the C++ cache transceiver uses, which moves KV point to point
between two known peers; the two compose.

A pool has one master, run as infrastructure rather than inside any engine::

    trtllm-serve mooncake_master --pool_file /shared/pool.json

Each server joins the pool the master's manifest describes::

    kv_connector_config:
      connector: mooncake-store
      mooncake_store:
        pool: file:///shared/pool.json
        role: both          # or: capacity, on a server that only lends memory
        segment_size: 160GiB

Every rank that joins contributes `segment_size`, so capacity is the sum over
participating ranks. `role` governs traffic only; see `config.StoreRole`.
`ledger.format_pool_report` totals up what each rank recorded.

Transferring requires `KVCacheManagerV2` and the Mooncake Python bindings,
which `tensorrt-llm` pulls in as `mooncake-transfer-engine-cuda13`; a
capacity-only rank needs only the bindings. An inherited `MOONCAKE_CONFIG_PATH`
wins over `mooncake_store`, so an externally managed pool stays reachable.
Registering the KV pools needs GPUDirect RDMA; where that is unavailable,
`stage_through_host: true` routes pages through a pinned host buffer.
"""

from .config import MooncakeStoreConnectorConfig, StoreRole, parse_size
from .ledger import SegmentRecord, format_pool_report, read_segments, record_segment
from .master import (
    POOL_MANIFEST_NAME,
    PoolManifest,
    local_address,
    maybe_provision_pool,
    provision_pool,
    resolve_device_name,
    resolve_pool,
    running_master,
    wait_for_master,
)
from .scheduler import MooncakeStoreConnectorScheduler
from .worker import MooncakeStoreConnectorWorker

__all__ = [
    "POOL_MANIFEST_NAME",
    "MooncakeStoreConnectorConfig",
    "MooncakeStoreConnectorScheduler",
    "MooncakeStoreConnectorWorker",
    "PoolManifest",
    "SegmentRecord",
    "StoreRole",
    "format_pool_report",
    "local_address",
    "maybe_provision_pool",
    "parse_size",
    "provision_pool",
    "read_segments",
    "record_segment",
    "resolve_device_name",
    "resolve_pool",
    "running_master",
    "wait_for_master",
]
