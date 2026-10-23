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
"""Running a Mooncake master for a test, on ports picked for it, and the memory behind it.

A deployment names its master's ports, so `running_master` takes them as
given. A test has no pair to name and wants any free one, on a machine whose
other ports belong to whatever else runs there. A port is reserved here by
binding it and letting go, so another process can still take it before the
master binds, and the master then exits before it accepts connections. Each
attempt picks a fresh pair.

The master stores nothing itself: `running_segment_provider` gives a pool its
memory from a separate process.
"""

import contextlib
import os
import subprocess  # nosec B404
import sys
import time
from typing import Iterator

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import local_address
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.master import (
    LaunchedMaster,
    resolve_device_name,
    running_master,
)
from tensorrt_llm._utils import get_free_ports

__all__ = ["running_master_on_free_ports", "running_segment_provider"]

PORT_ATTEMPTS = 5
SEGMENT_PROVIDER_TIMEOUT = 60.0

# Runs in the provider process: join the pool, mount the segment, and hold it until stdin closes.
_SEGMENT_PROVIDER = """
import pathlib, sys
from mooncake.store import MooncakeDistributedStore
master, segment_size, protocol, devices, hostname, ready = sys.argv[1:]
store = MooncakeDistributedStore()
code = store.setup(hostname, "P2PHANDSHAKE", int(segment_size), 16 << 20, protocol, devices, master)
if code != 0:
    sys.exit(f"segment provider: setup failed with {code}")
pathlib.Path(ready).touch()
sys.stdin.read()
store.close()
"""


def _start_master(
    stack: contextlib.ExitStack, run_dir: str, attempts: int, kwargs: dict
) -> LaunchedMaster:
    """Enter `running_master` on `stack`, on the first pair of ports it keeps.

    A failed start leaves nothing of the master behind, so every attempt
    begins from the state the first one did. The last failure is raised as it
    is, since it names the port that was taken.
    """
    for attempt in range(1, attempts + 1):
        rpc_port, metrics_port = get_free_ports(2)
        try:
            return stack.enter_context(
                running_master(run_dir, rpc_port=rpc_port, metrics_port=metrics_port, **kwargs)
            )
        except (RuntimeError, TimeoutError):
            if attempt == attempts:
                raise
            print(
                f"mooncake-store: the master did not come up on rpc={rpc_port} "
                f"metrics={metrics_port} (attempt {attempt} of {attempts}); "
                "retrying on another pair of ports."
            )
    raise AssertionError("unreachable: the last attempt either returns or raises")


@contextlib.contextmanager
def running_master_on_free_ports(
    run_dir: str, *, attempts: int = PORT_ATTEMPTS, **kwargs
) -> Iterator[LaunchedMaster]:
    """Run a master on free ports for the body's duration.

    Args:
        run_dir: Passed through to `running_master`.
        attempts: Pairs of ports to try.
        kwargs: Everything else `running_master` takes, except the two ports.
    """
    with contextlib.ExitStack() as stack:
        yield _start_master(stack, run_dir, attempts, kwargs)


@contextlib.contextmanager
def running_segment_provider(
    master_address: str,
    run_dir: str,
    *,
    segment_size: int,
    protocol: str = "tcp",
    device_name: str = "",
    timeout: float = SEGMENT_PROVIDER_TIMEOUT,
) -> Iterator[subprocess.Popen]:
    """Give the pool a `segment_size`-byte segment from a separate process, for the body's duration.

    A client with no segment of its own then finds every object in another
    process's memory, so each transfer goes through the transfer engine as it
    would across hosts, never through a copy inside one process.

    Args:
        master_address: The pool's master, `host:port`.
        run_dir: Where the provider's log and readiness marker go.
        segment_size: Bytes of the provider's segment.
        protocol: `rdma` or `tcp`.
        device_name: RDMA devices; empty picks the host's fastest InfiniBand ones.
        timeout: Seconds to wait for the provider to join the pool.
    """
    os.makedirs(run_dir, exist_ok=True)
    ready = os.path.join(run_dir, "segment_provider.ready")
    log_path = os.path.join(run_dir, "segment_provider.log")
    devices = resolve_device_name(protocol, device_name)
    command = [sys.executable, "-c", _SEGMENT_PROVIDER, master_address, str(segment_size)]
    command += [protocol, devices, local_address(master_address), ready]
    with open(log_path, "wb") as log:
        process = subprocess.Popen(  # nosec B603
            command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT
        )
    try:
        deadline = time.monotonic() + timeout
        while not os.path.exists(ready):
            if process.poll() is not None or time.monotonic() > deadline:
                with open(log_path, errors="replace") as log:
                    tail = log.read()[-4000:]
                raise RuntimeError(f"the segment provider did not join {master_address}:\n{tail}")
            time.sleep(0.1)
        yield process
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
