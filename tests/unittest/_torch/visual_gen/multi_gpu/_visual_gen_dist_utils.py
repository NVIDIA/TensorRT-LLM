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
"""Shared helpers for visual_gen multi-GPU (mp.spawn) tests.

Provides :func:`spawn_with_retry`, which allocates a fresh master port and
retries the spawn when the c10d rendezvous ``TCPStore`` loses the bind race and
fails with ``EADDRINUSE``.

Why the retry is needed: the CI-aware port allocator (``get_free_port_in_ci``)
binds a probe socket, *closes* it, and returns the port number. ``mp.spawn``
then launches fresh worker processes and rank 0's ``TCPStore`` re-binds that
port. Anything on the host (including ephemeral outbound sockets or a parallel
test) can grab the port in the gap between the probe close and the re-bind, so a
single allocation is not enough on busy nodes -- we must re-allocate and retry.
"""

import sys
from pathlib import Path

import torch.multiprocessing as mp

# The CI-aware allocator lives in tests/integration/defs/common.py. Adding that
# directory to sys.path lets us reuse it (it tracks allocated ports per-process
# so sequential tests don't collide, and honors CONTAINER_PORT_START/NUM).
_INTEGRATION_DIR = Path(__file__).resolve().parents[4] / "integration"
if str(_INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_DIR))
from defs.common import get_free_port_in_ci  # noqa: E402

_ADDR_IN_USE_MARKERS = ("EADDRINUSE", "address already in use")


def _is_addr_in_use(exc: BaseException) -> bool:
    msg = str(exc)
    return any(marker in msg for marker in _ADDR_IN_USE_MARKERS)


def spawn_with_retry(spawn_fn, max_retries: int = 10):
    """Run ``spawn_fn(port)`` with a fresh free port, retrying on EADDRINUSE.

    ``spawn_fn`` receives a master port and is expected to call ``mp.spawn``
    (passing the port through to the workers). If the rendezvous TCPStore fails
    to bind because the port was grabbed after allocation, a new port is chosen
    and the spawn is retried up to ``max_retries`` times. Any other failure
    (e.g. a real assertion inside the test) propagates immediately.
    """
    last_exc: BaseException | None = None
    for _ in range(max_retries):
        port = get_free_port_in_ci()
        try:
            spawn_fn(port)
            return
        except mp.ProcessRaisedException as exc:
            if _is_addr_in_use(exc):
                last_exc = exc
                continue
            raise
        except OSError as exc:
            if _is_addr_in_use(exc):
                last_exc = exc
                continue
            raise
    assert last_exc is not None
    raise last_exc
