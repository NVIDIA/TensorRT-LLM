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

Pool capacity comes only from processes that open a store handle: `setup`
registers `global_segment_size` bytes of the caller's host memory and the master
then places blocks in it. In a disaggregated deployment only the context servers
configure the connector, so the pool is entirely prefill-node memory, which
overlaps what TensorRT-LLM's own host offload already does.

Donating alongside a generation server puts that node's memory into the same
pool, so prefill writes blocks that land on decode-side DRAM. The generation
engine stays free of any connector and keeps its single cache transceiver for
the prefill-to-decode handoff.

Donation is not a `StoreRole`. The roles describe an engine's traffic and none
of them means "contribute memory only", so capacity and traffic stay separate
concerns and a donor holds a store handle of its own.

The memory is charged to the donating process, so size it together with that
node's `kv_cache_config.host_cache_size`.
"""

import contextlib
import time
from typing import Any, Iterator, Optional

from tensorrt_llm.logger import logger

from .config import DEFAULT_METADATA_SERVER, parse_size
from .master import (
    local_address,
    master_timeout,
    resolve_device_name,
    resolve_master_address,
    wait_for_master,
)

__all__ = [
    "DEFAULT_DONOR_LOCAL_BUFFER_SIZE",
    "donate_segment",
    "maybe_donate_segment",
]

#: A donor never transfers, but `setup` rejects a zero-sized transfer buffer.
DEFAULT_DONOR_LOCAL_BUFFER_SIZE = 64 * 1024**2


@contextlib.contextmanager
def donate_segment(
    master_server_address: str,
    segment_size: int,
    protocol: str = "rdma",
    device_name: str = "",
    metadata_server: str = DEFAULT_METADATA_SERVER,
    local_buffer_size: int = DEFAULT_DONOR_LOCAL_BUFFER_SIZE,
    hostname: Optional[str] = None,
) -> Iterator[str]:
    """Hold `segment_size` bytes of this node's memory in the pool.

    Yields the host the segment is registered under, which is how the master
    and the engines reading from it identify the capacity.

    Dropping the store handle unmounts the segment and the master starts
    reporting the blocks that lived in it as lost, so the caller must stay
    inside this context for as long as the capacity is meant to exist.
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
    donated = f"{segment_size / 1024**3:.1f}GiB"
    # Byte counts are spelled out next to the human-readable form. A misparsed
    # size string otherwise surfaces only as a pool that evicts far too eagerly.
    logger.info(
        f"mooncake-store: lending memory to the pool at {master_server_address} "
        f"as capacity only, no reads or writes: host={host} "
        f"segment_size={donated} ({segment_size} bytes) "
        f"protocol={protocol} device={device_name or '(none)'} "
        f"metadata_server={metadata_server} "
        f"local_buffer_size={local_buffer_size} bytes"
    )

    store = MooncakeDistributedStore()
    started = time.monotonic()
    status = store.setup(
        host,
        metadata_server,
        segment_size,
        local_buffer_size,
        protocol,
        device_name,
        master_server_address,
    )
    elapsed = time.monotonic() - started
    if status != 0:
        raise RuntimeError(
            f"Mooncake store.setup failed with status {status} after "
            f"{elapsed:.1f}s, so no memory was lent to the pool. The master at "
            f"{master_server_address} must already be accepting connections; "
            f"protocol={protocol!r} with device={device_name or '(none)'!r} "
            f"must be usable from {host}; and this node must have "
            f"{donated} of memory to spare, which it does not if its own "
            "kv_cache_config.host_cache_size has already claimed it."
        )

    logger.info(
        f"mooncake-store: {donated} of {host} is now part of the pool, "
        f"registered in {elapsed:.1f}s; the master at {master_server_address} "
        "can place blocks here from now on"
    )
    try:
        yield host
    finally:
        # The segment stays mounted while anything references the handle.
        del store
        logger.info(
            f"mooncake-store: withdrew the {donated} lent from {host}; the "
            "master will report blocks that lived there as lost"
        )


@contextlib.contextmanager
def maybe_donate_segment(donation: Any) -> Iterator[Optional[str]]:
    """Lend memory for this process's lifetime if the config asked to.

    Args:
        donation: A `MooncakeDonationConfig`, or `None` to do nothing, so
            callers need no condition of their own.

    Yields the host the segment is registered under, or `None`.
    """
    if donation is None:
        yield None
        return

    # Bringup blocks here on a master that may belong to a different job, so
    # name the address before waiting on it.
    logger.info(
        "mooncake-store: mooncake_donation is set, so this server lends host "
        f"memory to the pool at {donation.master_server_address} without using "
        "it; resolving the master now"
    )
    master_address = resolve_master_address(donation.master_server_address, master_timeout())
    # Checked before setup so an absent master is reported as such, rather than
    # as the status code setup returns for every kind of failure.
    wait_for_master(master_address)
    with donate_segment(
        master_server_address=master_address,
        segment_size=parse_size(donation.segment_size),
        protocol=donation.protocol,
        device_name=resolve_device_name(donation.protocol, donation.device_name),
        metadata_server=donation.metadata_server,
    ) as host:
        yield host
