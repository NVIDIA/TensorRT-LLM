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
"""Put a node's host memory into a Mooncake pool without reading or writing it.

Pool capacity comes only from processes that open a store handle: ``setup``
registers ``global_segment_size`` bytes of the caller's host memory and the
master then places blocks in it. In a disaggregated deployment only the context
servers configure the connector, so only they call ``setup``, and the pool is
entirely prefill-node memory -- which makes the store a
prefill-DRAM-caches-prefill-GPU tier, overlapping what TensorRT-LLM's own host
offload already does.

Donating alongside a generation server puts that node's memory into the same
pool. Prefill then writes blocks that land on decode-side DRAM and reads them
back, while the generation engine stays free of any connector: it neither reads
nor writes the store, so it keeps its single cache transceiver for the
prefill-to-decode handoff.

Donation is deliberately not a ``StoreRole``. The roles describe an engine's
traffic -- ``producer`` writes, ``consumer`` reads, ``both`` does both -- and
none of them means "contribute memory only", so attaching a connector to a
generation server to get its DRAM into the pool would also start it reading or
writing. Capacity and traffic are separate concerns, which is why this holds a
handle of its own rather than being a setting on the connector.

The memory is charged to the donating process, so it competes with anything
else on the node -- a generation server's ``kv_cache_config.host_cache_size``
above all. Size the two together.
"""

import contextlib
from typing import Iterator, Optional

from tensorrt_llm.logger import logger

from .master import local_address

__all__ = ["DEFAULT_DONOR_LOCAL_BUFFER_SIZE", "donate_segment"]

#: A donor never transfers, so its transfer buffer is dead weight; ``setup``
#: still rejects a zero one.
DEFAULT_DONOR_LOCAL_BUFFER_SIZE = 64 * 1024**2


@contextlib.contextmanager
def donate_segment(
    master_server_address: str,
    segment_size: int,
    protocol: str = "rdma",
    device_name: str = "",
    metadata_server: str = "",
    local_buffer_size: int = DEFAULT_DONOR_LOCAL_BUFFER_SIZE,
    hostname: Optional[str] = None,
) -> Iterator[str]:
    """Hold ``segment_size`` bytes of this node's memory in the pool.

    Yields the host the segment is registered under, which is what the master
    and the engines reading from it identify the capacity by.

    The handle is held for the duration: dropping it unmounts the segment, and
    the master starts reporting the blocks that lived in it as lost. So the
    caller must stay inside this context for as long as the capacity is meant
    to exist, which for a donor is its whole run.
    """
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as exc:
        raise ImportError(
            "Donating memory needs the Mooncake Python bindings "
            "(`pip install mooncake-transfer-engine`). The C++ transfer engine "
            f"in the container is a different component: {exc}"
        ) from exc

    host = hostname or local_address()
    logger.info(
        f"mooncake-store: joining the pool at {master_server_address} as "
        f"capacity only: host={host} protocol={protocol} "
        f"device={device_name or '(none)'} "
        f"donating={segment_size / 1024 ** 3:.1f}GiB"
    )

    store = MooncakeDistributedStore()
    status = store.setup(
        host,
        metadata_server,
        segment_size,
        local_buffer_size,
        protocol,
        device_name,
        master_server_address,
    )
    if status != 0:
        raise RuntimeError(
            f"Mooncake store.setup failed with status {status}. The master at "
            f"{master_server_address} must already be accepting connections, "
            f"and protocol={protocol!r} must be usable from {host}."
        )

    logger.info(
        f"mooncake-store: {segment_size / 1024 ** 3:.1f}GiB from {host} is now "
        "part of the pool"
    )
    try:
        yield host
    finally:
        # Explicit because the segment stays mounted for as long as anything
        # references the handle, and "as long as this context" is the contract.
        del store
        logger.info(f"mooncake-store: withdrew the segment donated from {host}")
