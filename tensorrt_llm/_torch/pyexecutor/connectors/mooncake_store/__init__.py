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

Requires `KVCacheManagerV2`, the manager that can describe its pools to a
connector through `register_kv_cache_layout`, and the Mooncake Python bindings
(`pip install mooncake-transfer-engine`).

Enable it with::

    kv_connector_config = KvCacheConnectorConfig(connector="mooncake-store")

with `MOONCAKE_CONFIG_PATH` pointing at a Mooncake JSON config. Describing the
pool in `KvCacheConnectorConfig.mooncake_store` instead lets `trtllm-serve`
provision it during bringup, so no external script has to; see `master.py`.

Capacity comes only from processes that open a store handle, which in a
disaggregated deployment is the context servers alone. `donor.py` lends a
node's memory to the pool without giving it a connector.

By default the KV pools themselves are registered with Mooncake, which requires
GPUDirect RDMA. Where that is unavailable, `"stage_through_host": true` in the
JSON config, or `TRTLLM_MOONCAKE_STORE_STAGE_THROUGH_HOST=1`, routes pages
through a pinned host buffer instead; see `staging.py`.
"""

from .config import MooncakeStoreConnectorConfig, StoreRole, parse_size
from .donor import (DEFAULT_DONOR_LOCAL_BUFFER_SIZE, donate_segment,
                    maybe_donate_segment)
from .master import (local_address, master_timeout, maybe_provision_pool,
                     provision_pool, resolve_device_name,
                     resolve_master_address, running_master, wait_for_master)
from .scheduler import MooncakeStoreConnectorScheduler
from .worker import MooncakeStoreConnectorWorker

__all__ = [
    "DEFAULT_DONOR_LOCAL_BUFFER_SIZE",
    "MooncakeStoreConnectorConfig",
    "MooncakeStoreConnectorScheduler",
    "MooncakeStoreConnectorWorker",
    "StoreRole",
    "donate_segment",
    "local_address",
    "master_timeout",
    "maybe_donate_segment",
    "maybe_provision_pool",
    "parse_size",
    "provision_pool",
    "resolve_device_name",
    "resolve_master_address",
    "running_master",
    "wait_for_master",
]
