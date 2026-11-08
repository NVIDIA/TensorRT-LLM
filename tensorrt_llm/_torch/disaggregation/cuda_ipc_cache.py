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
"""Bound the UCX CUDA IPC remote-handle cache used by KV cache transfers.

A KV sender that writes over UCX ``cuda_ipc`` imports and maps each peer
allocation it writes into, and UCX keeps that mapping in a per-peer cache. The
transfer agents register a peer's KV cache pools per physical VMM chunk (up to
32 MiB each), so a sender accumulates one mapping per peer chunk it has ever
written. UCX leaves this cache unbounded by default, so it grows over a long
run until the sender has mapped every chunk of every peer.

Like every device virtual-memory mapping (``cuMemMap`` + ``cuMemSetAccess``,
local or imported), each of these takes an entry in a driver table shared by
all processes and GPUs on the node (about 2**19 entries). When that table is
full, every new mapping on the node fails with CUDA_ERROR_OUT_OF_MEMORY (UCX
imports, PyTorch expandable-segment growth, lazy kernel loads) while device
memory is still free. Bounding the per-peer cache keeps a sender's share of
that table fixed; an evicted chunk is imported again on its next write.
"""

import os
from collections.abc import MutableMapping

from tensorrt_llm.logger import logger

CUDA_IPC_CACHE_MAX_REGIONS_ENV = "UCX_CUDA_IPC_CACHE_MAX_REGIONS"
CUDA_IPC_CACHE_MAX_SIZE_ENV = "UCX_CUDA_IPC_CACHE_MAX_SIZE"

# Mapped peer regions kept per peer process. With 32 MiB chunks this is 16 GiB of
# a peer's KV cache, and a sender writing to 16 peers holds at most 8192 mappings.
DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS = 512


def bound_cuda_ipc_cache(environ: MutableMapping[str, str] | None = None) -> str | None:
    """Default UCX's per-peer CUDA IPC cache to a bounded number of regions.

    UCX reads the limit when a transfer agent opens its ``cuda_ipc`` memory
    domain and keeps the tightest value seen in the process, so this must run
    before the first transfer agent is created. A limit the user already set
    through ``UCX_CUDA_IPC_CACHE_MAX_REGIONS`` or ``UCX_CUDA_IPC_CACHE_MAX_SIZE``
    (including ``inf``) is left unchanged.

    Args:
        environ: Environment to update; the process environment if None.

    Returns:
        The region limit this call set, or None if the user configured one.
    """
    env = os.environ if environ is None else environ
    if CUDA_IPC_CACHE_MAX_REGIONS_ENV in env or CUDA_IPC_CACHE_MAX_SIZE_ENV in env:
        logger.info(
            "UCX CUDA IPC remote-handle cache limit set by the user: "
            f"{CUDA_IPC_CACHE_MAX_REGIONS_ENV}={env.get(CUDA_IPC_CACHE_MAX_REGIONS_ENV, '<unset>')} "
            f"{CUDA_IPC_CACHE_MAX_SIZE_ENV}={env.get(CUDA_IPC_CACHE_MAX_SIZE_ENV, '<unset>')}"
        )
        return None
    value = str(DEFAULT_CUDA_IPC_CACHE_MAX_REGIONS)
    env[CUDA_IPC_CACHE_MAX_REGIONS_ENV] = value
    logger.info(
        f"Bounding the UCX CUDA IPC remote-handle cache to {value} regions per peer "
        f"({CUDA_IPC_CACHE_MAX_REGIONS_ENV}={value}; set it to 'inf' to keep every mapping)."
    )
    return value
