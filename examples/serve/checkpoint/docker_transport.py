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
"""One-shot stdio helper executed by the trusted image's isolated Python interpreter."""

import http.client
import json
import signal
import struct
import sys
import time
from contextlib import closing

_ROUTES = {
    "/checkpoint/status": "GET",
    "/checkpoint/prepare_capture": "POST",
    "/checkpoint/validate_runtime": "POST",
    "/checkpoint/activate": "POST",
    "/ready": "GET",
    "/v1/completions": "POST",
}
_MAX_FRAME = 128 * 1024
_AUTH_HEADERS = {
    "x-trtllm-runtime-control-auth",
    "x-trtllm-runtime-control-nonce",
    "x-trtllm-runtime-control-timestamp",
}


def _run(port: int, path: str, startup_budget: float) -> None:
    method = _ROUTES[path]
    if not 1 <= port <= 65535 or not 0 < startup_budget <= 3600:
        raise ValueError("Invalid endpoint or budget.")
    # Killing a detached Docker CLI does not necessarily kill the exec in the container.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, startup_budget)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=startup_budget)
    conn.auto_open = 0
    with closing(conn):
        conn.connect()
        connected_at = time.monotonic()
        out = sys.stdout.buffer
        out.write(b"R")
        out.flush()
        frame = sys.stdin.buffer.readline(_MAX_FRAME + 1)
        if len(frame) > _MAX_FRAME or not frame.endswith(b"\n"):
            raise ValueError("Incomplete or oversized request.")
        ready_budget, headers, body = json.loads(frame)
        allowed = _AUTH_HEADERS if path.startswith("/checkpoint/") else set()
        if method == "POST":
            allowed = allowed | {"Content-Type"}
        if set(headers) != allowed or bool(body) != (method == "POST"):
            raise ValueError("Unexpected checkpoint request.")
        remaining = connected_at + ready_budget - time.monotonic()
        if not 0 < remaining <= startup_budget:
            raise TimeoutError("Expired request.")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        conn.request(method, path, body=body.encode("ascii") or None, headers=headers)
        with conn.getresponse() as response:
            status = response.status
            raw = response.read(_MAX_FRAME + 1) if status == 200 else b""
            if len(raw) > _MAX_FRAME or (status == 200 and response.length not in (None, 0)):
                raise ValueError("Incomplete or oversized response.")
    out.write(struct.pack("!HI", status, len(raw)) + raw)
    out.flush()


if __name__ == "__main__":
    _run(int(sys.argv[1]), sys.argv[2], float(sys.argv[3]))
