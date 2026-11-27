# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper process that reports one MPI worker's liveness to its proxy.

WorkerProcessWatch runs this file by path in a separate interpreter. It imports
only the standard library, pyzmq and the sibling framing module, never
TensorRT-LLM, CUDA or MPI, so it keeps reporting while the worker is blocked.
"""

from __future__ import annotations

import json
import os

# Only serializes reports; the proxy authenticates frames before deserialization.
import pickle  # nosec B403
import selectors
import socket
import sys
import time

import zmq

if __package__:
    from .hmac_framing import sign_frame
else:
    # Run by path, this directory is on sys.path and the package is not loaded.
    from hmac_framing import sign_frame

HEARTBEAT_INTERVAL = 0.5
# How long the worker waits for the helper's first report to be sent.
START_TIMEOUT = 10.0


def wait_readable(stream, timeout: float) -> bool:
    """Wait until a pipe is readable without select()'s FD_SETSIZE limit.

    Workers can hold more than 1024 descriptors (CUDA, NCCL, sockets), so the
    helper's pipe may be numbered beyond what select.select() accepts.
    """
    with selectors.DefaultSelector() as selector:
        selector.register(stream, selectors.EVENT_READ)
        return bool(selector.select(timeout))


def main() -> int:
    """Report until the worker stops the watch or exits.

    Reads one JSON configuration line (address, hex key, rank, worker pid,
    ipv6) from stdin, prints "ready" after the first report is queued, and
    then sends a report every HEARTBEAT_INTERVAL. A "stop" line ends the watch
    with a "returned" report; end of input or a changed parent ends it with an
    "exited" report.

    Returns:
        0 after the final report, 1 if the first report could not be sent
        within START_TIMEOUT.
    """
    config = json.loads(sys.stdin.readline())
    key = bytes.fromhex(config.pop("key"))
    report = {
        "rank": config["rank"],
        "pid": config["pid"],
        "host": socket.gethostname(),
        "sequence": 0,
        "kind": "alive",
    }
    with zmq.Context() as context, context.socket(zmq.PUSH) as channel:
        channel.setsockopt(zmq.IPV6, int(config.get("ipv6", False)))
        # Queue reports only on a connected peer, so "ready" proves the connection.
        channel.setsockopt(zmq.IMMEDIATE, 1)
        channel.setsockopt(zmq.LINGER, 250)
        channel.setsockopt(zmq.SNDHWM, 4)
        channel.setsockopt(zmq.SNDTIMEO, 100)
        channel.connect(config["address"])
        # Leave the worker time to read "ready" before its own START_TIMEOUT.
        deadline = time.monotonic() + START_TIMEOUT - 1
        ready = False
        while True:
            if os.getppid() != config["pid"]:
                report["kind"] = "exited"
            try:
                channel.send(sign_frame(key, pickle.dumps(report)))
                if not ready:
                    print("ready", flush=True)
                    ready = True
            except zmq.Again:
                if not ready and time.monotonic() >= deadline:
                    return 1
            if report["kind"] != "alive":
                return 0
            report["sequence"] += 1
            if wait_readable(sys.stdin, HEARTBEAT_INTERVAL):
                # A stop request means the worker call ended in a live process;
                # end of input without one means the worker process is gone.
                stopped = sys.stdin.readline().strip() == "stop"
                report["kind"] = "returned" if stopped else "exited"


if __name__ == "__main__":
    if sys.argv[1:] != ["--watch"]:
        raise SystemExit("Expected --watch")
    raise SystemExit(main())
