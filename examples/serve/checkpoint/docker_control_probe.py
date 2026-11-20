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
"""Read-only probe executed inside the private workload PID and network namespaces."""

import json
import os
import signal
import socket
import sys
from pathlib import Path


def verify_listener(proc: Path, port: int, restored_pid: int | None) -> None:
    """Require an unambiguous private IPv4 listener, owned by the restored tree if specified."""
    sockets = []
    for table in ("tcp", "tcp6"):
        path = proc / "net" / table
        if table == "tcp6" and not path.exists():
            continue
        for line in path.read_text().splitlines()[1:]:
            fields = line.split()
            address, number = fields[1].split(":")
            if fields[3] == "0A" and int(number, 16) == port:
                sockets.append((table, address, fields[9]))
    if len(sockets) != 1 or sockets[0][:2] != ("tcp", "0100007F"):
        raise ValueError("Expected exactly one IPv4 loopback control listener")
    if restored_pid is None:
        return
    socket_link = f"socket:[{sockets[0][2]}]"
    owners = set()
    parents = {}
    for task in proc.iterdir():
        if not task.name.isdecimal():
            continue
        pid = int(task.name)
        try:
            # comm can contain spaces and parentheses; state and ppid follow the last ')'.
            parents[pid] = int((task / "stat").read_text().rsplit(")", 1)[1].split()[1])
            for fd in (task / "fd").iterdir():
                try:
                    if os.readlink(fd) == socket_link:
                        owners.add(pid)
                except FileNotFoundError:
                    continue
        except FileNotFoundError:
            continue
    if not owners or restored_pid not in parents:
        raise ValueError("Restored process or listener owner absent")
    for owner in owners:
        seen = set()
        while owner != restored_pid:
            if owner not in parents or owner in seen:
                raise ValueError("Control listener is not owned by the restored process tree")
            seen.add(owner)
            owner = parents[owner]


def main() -> None:
    """Print only observed bind identity and GPU selector, never workload credentials."""
    pid, budget, port = sys.argv[1:]
    info = os.stat("/snapshot-control")
    if budget:
        signal.setitimer(signal.ITIMER_REAL, float(budget))
        with socket.create_connection(("127.0.0.1", int(port)), float(budget)):
            pass
        verify_listener(Path("/proc"), int(port), int(pid) if pid else None)
    print(
        json.dumps(
            [
                info.st_dev,
                info.st_ino,
                os.environ.get("SNAPSHOT_CONTROL_DIR"),
                os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            ]
        )
    )


if __name__ == "__main__":
    main()
