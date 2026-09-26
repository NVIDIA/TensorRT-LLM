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
"""A Mooncake distributed store to back a KV cache connector.

The store is a shared CPU memory pool addressed by content, so a prefix computed
by one engine can be replayed by another, which regular block reuse cannot do
because it never leaves the instance that computed it.

This is a different component from the Mooncake transfer engine that the C++
cache transceiver uses for disaggregated prefill/decode handoff: that moves KV
point to point between two known peers, while this one publishes pages into a
pool addressed by content. The two compose, so a context server can write pages
here and still hand off over NIXL.

`master.py` brings a pool up: it resolves or launches the `mooncake_master`
and renders the client config the workers read. Capacity comes only from
processes that open a store handle, which in a disaggregated deployment is the
context servers alone, so `donor.py` lends a node's memory to the pool without
giving it a connector. `trtllm-serve mooncake_master` and `mooncake_donor`
expose both. Both need the Mooncake Python bindings, an optional dependency
installed with `pip install tensorrt-llm[mooncake]`.

`keys.py` and `staging.py` hold what the store side shares with the connector
that moves pages in and out of the pool: how a block of tokens becomes a store
key, and how pages reach the fabric on hosts without GPUDirect RDMA. The
connector itself, and the `LlmArgs` surface that selects it, land with the KV
cache manager V2 support it depends on.
"""

from .config import MooncakeStoreConnectorConfig, StoreRole, parse_size
from .donor import DEFAULT_DONOR_LOCAL_BUFFER_SIZE, donate_segment
from .master import (
    PoolSpec,
    local_address,
    master_timeout,
    provision_pool,
    resolve_device_name,
    resolve_master_address,
    running_master,
    wait_for_master,
    write_client_config,
)

__all__ = [
    "DEFAULT_DONOR_LOCAL_BUFFER_SIZE",
    "MooncakeStoreConnectorConfig",
    "PoolSpec",
    "StoreRole",
    "donate_segment",
    "local_address",
    "master_timeout",
    "parse_size",
    "provision_pool",
    "resolve_device_name",
    "resolve_master_address",
    "running_master",
    "wait_for_master",
    "write_client_config",
]
