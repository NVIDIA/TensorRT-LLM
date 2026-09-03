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
import time
from typing import Any, Iterator, Optional

from tensorrt_llm.logger import logger

from .config import parse_size
from .master import (local_address, master_timeout, resolve_device_name,
                     resolve_master_address, wait_for_master)

__all__ = [
    "DEFAULT_DONOR_LOCAL_BUFFER_SIZE",
    "donate_segment",
    "maybe_donate_segment",
]

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
    donated = f"{segment_size / 1024 ** 3:.1f}GiB"
    # Every argument is echoed, with the byte counts spelled out next to the
    # human-readable form: a segment that is a thousandth of the intended size
    # is a size string parsed wrong, and it otherwise shows up only as a pool
    # that evicts far too eagerly, days later.
    logger.info(
        f"mooncake-store: lending memory to the pool at {master_server_address} "
        f"as capacity only, no reads or writes: host={host} "
        f"segment_size={donated} ({segment_size} bytes) "
        f"protocol={protocol} device={device_name or '(none)'} "
        f"metadata_server={metadata_server or '(none)'} "
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
        # Explicit because the segment stays mounted for as long as anything
        # references the handle, and "as long as this context" is the contract.
        del store
        logger.info(
            f"mooncake-store: withdrew the {donated} lent from {host}; the "
            "master will report blocks that lived there as lost"
        )


@contextlib.contextmanager
def maybe_donate_segment(donation: Any) -> Iterator[Optional[str]]:
    """Lend memory for this process's lifetime if the config asked to.

    Args:
        donation: A ``MooncakeDonationConfig``, or ``None`` to do nothing --
            which is every deployment that does not lend memory, so callers
            need no condition of their own.

    Yields the host the segment is registered under, or ``None``.
    """
    if donation is None:
        yield None
        return

    # A generation server has no other reason to resolve a master, so the
    # address it lends against is worth saying out loud before the wait: this
    # is the one place bringup blocks on a component from a different job.
    logger.info(
        "mooncake-store: mooncake_donation is set, so this server lends host "
        f"memory to the pool at {donation.master_server_address} without using "
        "it; resolving the master now"
    )
    master_address = resolve_master_address(
        donation.master_server_address, master_timeout()
    )
    # Checked before setup so an absent master reads as one, rather than as
    # the status code setup returns for everything.
    wait_for_master(master_address)
    with donate_segment(
        master_server_address=master_address,
        segment_size=parse_size(donation.segment_size),
        protocol=donation.protocol,
        # Which HCAs this node has is the node's business, so a config that
        # leaves it open stays usable on every node type in the deployment.
        device_name=resolve_device_name(donation.protocol, donation.device_name),
        metadata_server=donation.metadata_server,
    ) as host:
        yield host
