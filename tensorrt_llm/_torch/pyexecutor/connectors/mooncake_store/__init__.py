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
pool addressed by content. The two compose, so a context server can write pages
here and still hand off over NIXL.

Transferring requires `KVCacheManagerV2`, the manager that can describe its
pools to a connector through `register_kv_cache_layout`, and the Mooncake Python
bindings, which `tensorrt-llm` pulls in as `mooncake-transfer-engine-cuda13`.
A capacity-only rank describes no pools, so it needs only the bindings.

A pool has one master, run as infrastructure rather than inside any engine::

    trtllm-serve mooncake_master --pool_file /shared/pool.json

It publishes a manifest describing the pool, which each server then joins::

    kv_connector_config:
      connector: mooncake-store
      mooncake_store:
        pool: file:///shared/pool.json
        role: both          # or: capacity, on a server that only lends memory
        segment_size: 160GiB

Every rank that joins contributes `segment_size`, so capacity is the sum over
participating ranks and grows with the deployment's parallelism. `role` governs
only traffic, which is what lets a generation server lend its memory while
leaving the pool alone; see `config.StoreRole`. What each rank contributed is
recorded under the run directory for `ledger.format_pool_report` to total up.

Pointing `MOONCAKE_CONFIG_PATH` at a Mooncake JSON config directly also works,
and wins over `mooncake_store`, so an externally managed pool stays reachable.

By default the KV pools themselves are registered with Mooncake, which requires
GPUDirect RDMA. Where that is unavailable, `stage_through_host: true` routes
pages through a pinned host buffer instead; see `staging.py`.
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
