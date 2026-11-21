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
"""Lend a node's host memory to a Mooncake store pool without serving from it.

Pool capacity comes only from processes that hold a store handle: `setup`
mounts the caller's segment of host memory and the master places blocks in it.
A node whose servers run no KV connector, such as a generation node in a
disaggregated deployment, therefore adds nothing to the pool.
`trtllm-serve mooncake_donor` runs beside such a server and mounts a segment of
that node's memory, so the context servers' blocks can land there while the
generation engine stays free of any connector.

A server with `mooncake_store.role: capacity` lends memory too, but only by
attaching a connector to its engine. A donor holds the segment and nothing
else: no KV cache is registered and no page moves through it.
"""

import contextlib
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from tensorrt_llm.logger import logger

from .config import local_address
from .ledger import DONOR_ROLE, record_segment
from .master import (
    DEFAULT_MASTER_TIMEOUT,
    PoolManifest,
    resolve_device_name,
    resolve_pool,
    wait_for_master,
)
from .validation import validate_donation_budget

__all__ = ["DEFAULT_DONOR_LOCAL_BUFFER_SIZE", "DonatedSegment", "donate_segment"]

#: Mooncake's per-process transfer buffer. A donor transfers nothing, but
#: `setup` refuses a buffer of zero bytes.
DEFAULT_DONOR_LOCAL_BUFFER_SIZE = 64 * 1024**2

_GIB = 1 << 30


@dataclass(frozen=True)
class DonatedSegment:
    """A segment of this node's memory that the pool holds for now."""

    #: Host the segment is registered under, as the master knows it.
    host: str
    segment_size: int
    #: The pool it was lent to.
    manifest: PoolManifest
    #: RDMA devices it was registered on; empty when Mooncake chose them.
    device_name: str


def _close_after_failed_setup(store) -> None:
    """Close a store whose setup failed, without hiding that failure."""
    try:
        store.close()
    except Exception as exc:  # noqa: BLE001 - the setup failure is the one to report
        logger.error(f"mooncake-store: closing the store after a failed setup also failed: {exc!r}")


@contextlib.contextmanager
def donate_segment(
    pool: str,
    segment_size: int,
    *,
    device_name: str = "",
    local_hostname: Optional[str] = None,
    timeout: float = DEFAULT_MASTER_TIMEOUT,
    run_dir: Optional[str] = None,
    local_buffer_size: int = DEFAULT_DONOR_LOCAL_BUFFER_SIZE,
) -> Iterator[DonatedSegment]:
    """Hold `segment_size` bytes of this node's memory in the pool `pool` names.

    Args:
        pool: The pool, named as `mooncake_store.pool` names it:
            `file://<path>` for the manifest `trtllm-serve mooncake_master
            --pool_file` publishes, or a master's bare `host:port`.
        segment_size: Bytes of host memory to mount. Must be positive.
        device_name: Comma-separated RDMA devices to register the segment on.
            Empty detects them as a server's provisioning does; see
            `master.resolve_device_name`.
        local_hostname: Address to register the segment under. Defaults to
            the interface that routes to the master.
        timeout: Seconds to wait for the manifest to appear and for the master
            to accept connections.
        run_dir: Where to record the segment for `trtllm-serve
            mooncake_pool_report`. Nothing is recorded without one.
        local_buffer_size: Mooncake's per-process transfer buffer, in bytes.

    Yields:
        The segment, mounted for as long as the context lasts. Leaving the
        context unmounts it, and the master then reports the blocks that lived
        there as lost.
    """
    if segment_size <= 0:
        raise ValueError(f"A donor lends memory, but segment_size is {segment_size} bytes")
    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as exc:
        raise ImportError(
            "Lending memory to a Mooncake pool needs the Mooncake Python "
            "bindings, which `tensorrt-llm` pulls in as "
            "`mooncake-transfer-engine-cuda13`, so reinstall that package if "
            "this environment dropped it."
        ) from exc

    manifest = resolve_pool(pool, timeout)
    # Before setup, so an absent master is reported as such rather than as the
    # status code setup returns for every kind of failure.
    wait_for_master(manifest.master_server_address, timeout)
    devices = resolve_device_name(manifest.protocol, device_name)
    host = local_hostname or local_address(manifest.master_server_address)
    lent = f"{segment_size / _GIB:.1f} GiB"
    # The byte count is spelled out beside the rounded size: a misread size
    # otherwise shows up only as a pool that evicts far too early.
    logger.warning(
        f"mooncake-store: lending {lent} ({segment_size} bytes) of {host} to the "
        f"pool at {manifest.describe()}, devices={devices or '(Mooncake chooses)'}"
    )

    # Before setup allocates anything, so an oversized donation fails here
    # instead of reaching the OOM killer.
    validate_donation_budget(segment_size, local_buffer_size)
    store = MooncakeDistributedStore()
    started = time.monotonic()
    try:
        status = store.setup(
            host,
            manifest.metadata_server,
            segment_size,
            local_buffer_size,
            manifest.protocol,
            devices,
            manifest.master_server_address,
        )
    except BaseException:
        _close_after_failed_setup(store)
        raise
    elapsed = time.monotonic() - started
    if status != 0:
        # A failed setup can hold part of what it allocated until closed.
        _close_after_failed_setup(store)
        raise RuntimeError(
            f"MooncakeDistributedStore.setup failed with status {status} after "
            f"{elapsed:.1f}s, so nothing was lent to the pool at "
            f"{manifest.master_server_address}. protocol={manifest.protocol!r} "
            f"with devices={devices!r} must be usable from {host}, and this node "
            f"must have {lent} of host memory to spare beside what its servers "
            "already hold."
        )
    try:
        logger.warning(
            f"mooncake-store: {lent} of {host} joined the pool in {elapsed:.1f}s; "
            "the master can place blocks in it from now on"
        )
        record_segment(
            run_dir,
            host=host,
            rank=0,
            segment_size=segment_size,
            role=DONOR_ROLE,
            model_key="",
        )
        yield DonatedSegment(
            host=host, segment_size=segment_size, manifest=manifest, device_name=devices
        )
    finally:
        try:
            store.close()
        except BaseException:
            logger.error(
                f"mooncake-store: closing the store that holds the {lent} {host} lent "
                "failed, so the master may still count that segment"
            )
            raise
        logger.warning(
            f"mooncake-store: withdrew the {lent} {host} lent; the master reports "
            "the blocks that lived there as lost"
        )
