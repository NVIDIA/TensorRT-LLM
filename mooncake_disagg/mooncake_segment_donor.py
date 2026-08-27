#!/usr/bin/env python3
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
"""Donate this node's host memory to a Mooncake store pool, without reading or
writing it.

Pool capacity comes only from processes that open a store handle: ``setup``
registers ``global_segment_size`` bytes of the calling process's host memory and
the master then places blocks in it. In a disaggregated deployment only the
context servers configure the KV connector, so only they call ``setup``, and the
pool is entirely prefill-node memory -- which makes the store a
prefill-DRAM-caches-prefill-GPU tier, overlapping what TensorRT-LLM's native
host offload already does.

Running this alongside a generation server puts that node's memory into the same
pool. Prefill then writes blocks that land on decode-side DRAM, and reads them
back, while the generation engine itself stays free of any connector: it neither
reads nor writes the store, so it keeps its single cache transceiver for the
prefill-to-decode KV handoff.

A donor is deliberately not a ``StoreRole``. The roles describe an engine's
traffic (``producer`` writes, ``consumer`` reads, ``both``), and none of them
means "contribute memory only" -- attaching a connector to the generation server
to get its DRAM into the pool would also start it reading or writing. Capacity
and traffic are separate concerns, so donation is a separate process.

The donated memory is charged to this process, so it competes with the
generation server's own ``kv_cache_config.host_cache_size`` on the same node.
Size the two together.
"""

import argparse
import json
import os
import re
import signal
import sys
import threading
import time

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "m": 1000**2,
    "mb": 1000**2,
    "g": 1000**3,
    "gb": 1000**3,
    "t": 1000**4,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}
_SIZE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$")

# The donor never transfers, so its staging buffer is dead weight; setup still
# rejects a zero one.
DEFAULT_LOCAL_BUFFER_SIZE = "64MiB"


def parse_size(value) -> int:
    """Accept either a byte count or a suffixed string such as ``"32GiB"``.

    Mirrors the connector's parser so a size means the same thing in
    ``mooncake.json`` and on this script's command line.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected a size, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = _SIZE_RE.match(str(value))
    if match is None:
        raise ValueError(f"cannot parse size {value!r}")
    magnitude, unit = match.groups()
    scale = _SIZE_UNITS.get(unit.lower())
    if scale is None:
        raise ValueError(f"unknown size unit {unit!r} in {value!r}")
    return int(float(magnitude) * scale)


def log(message: str) -> None:
    print(f"[donor {time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.getenv("MOONCAKE_CONFIG_PATH"),
        help="Mooncake JSON config naming the pool to join. Defaults to "
        "$MOONCAKE_CONFIG_PATH.",
    )
    parser.add_argument(
        "--segment-size",
        default="32GiB",
        help="Host memory to contribute, e.g. 32GiB. Overrides the config's "
        "global_segment_size, which is sized for an engine worker rather "
        "than a node donating spare memory.",
    )
    parser.add_argument(
        "--ready-file",
        default=None,
        help="File to create once the segment is mounted, for launchers that "
        "must not start writing to the pool before it has this capacity.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=300,
        help="Interval between liveness lines. 0 disables them.",
    )
    args = parser.parse_args()

    if not args.config:
        parser.error("--config is required when MOONCAKE_CONFIG_PATH is unset")

    with open(args.config) as handle:
        raw = json.load(handle)

    master = raw.get("master_server_address", "")
    if not master:
        parser.error(f"{args.config} has no master_server_address")

    segment_size = parse_size(args.segment_size)
    local_buffer_size = parse_size(
        raw.get("local_buffer_size_donor", DEFAULT_LOCAL_BUFFER_SIZE)
    )
    protocol = raw.get("protocol", "rdma")
    device_name = raw.get("device_name", "") or ""
    metadata_server = raw.get("metadata_server", "")

    try:
        from mooncake.store import MooncakeDistributedStore
    except ImportError as exc:
        log(
            "the Mooncake Python bindings are missing "
            "(`pip install mooncake-transfer-engine`); the C++ transfer engine "
            f"in the container is a different component: {exc}"
        )
        return 1

    import socket

    hostname = socket.gethostbyname(socket.gethostname())

    log(
        f"joining pool at {master} as a capacity-only client: "
        f"host={hostname} protocol={protocol} device={device_name or '(none)'} "
        f"donating={segment_size / 1024 ** 3:.1f}GiB"
    )

    # Held for the process's lifetime: dropping the handle unmounts the segment
    # and the master starts reporting the blocks living in it as lost.
    store = MooncakeDistributedStore()
    status = store.setup(
        hostname,
        metadata_server,
        segment_size,
        local_buffer_size,
        protocol,
        device_name,
        master,
    )
    if status != 0:
        log(
            f"setup failed with status {status}. The master must already be "
            f"accepting connections at {master}, and protocol={protocol!r} "
            "must be usable from this node."
        )
        return 1

    log(f"segment mounted; {segment_size / 1024 ** 3:.1f}GiB now available to the pool")

    if args.ready_file:
        with open(args.ready_file, "w") as handle:
            handle.write(f"{hostname} {segment_size}\n")

    stop = threading.Event()

    def handle_signal(signum, _frame):
        log(f"received signal {signum}; unmounting segment")
        stop.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Idle by design. Any put or get here would make this node a store client in
    # the traffic sense, which is what keeping the generation engine
    # connector-free is meant to avoid.
    heartbeat = args.heartbeat_seconds
    started = time.monotonic()
    while not stop.is_set():
        if heartbeat > 0:
            if stop.wait(heartbeat):
                break
            log(f"alive, donating for {(time.monotonic() - started) / 60:.0f}m")
        else:
            stop.wait()

    del store
    log("exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
