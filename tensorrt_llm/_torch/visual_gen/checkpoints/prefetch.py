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
"""Host page-cache prefetch helpers for visual generation checkpoints."""

import multiprocessing
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional, Set

import psutil
import torch.distributed as dist

from tensorrt_llm.logger import logger

_PREFETCH_CHUNK_SIZE = 16 * 1024 * 1024


def _dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _dist_barrier() -> None:
    if _dist_initialized():
        dist.barrier()


def _local_prefetch_context() -> tuple[int, int, int]:
    """Discover node peers and their minimum available memory on every call.

    Host page-cache sharing follows host membership, not the number of visible
    GPUs or global ranks. Gathering host identities also supports interleaved
    ranks and launchers that do not export LOCAL_WORLD_SIZE.
    """
    available_memory = int(psutil.virtual_memory().available)
    if not _dist_initialized() or dist.get_world_size() == 1:
        return 0, 1, available_memory

    hostname = socket.gethostname()
    gathered: list[tuple[str, int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (hostname, available_memory))
    local_ranks = [
        rank for rank, entry in enumerate(gathered) if entry is not None and entry[0] == hostname
    ]
    local_memory = [entry[1] for entry in gathered if entry is not None and entry[0] == hostname]
    return local_ranks.index(dist.get_rank()), len(local_ranks), min(local_memory)


def _normalize_paths(
    file_names: Iterable[str],
    prefetched_paths: Optional[Set[str]],
) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for file_name in file_names:
        path = os.path.abspath(file_name)
        if path in seen:
            continue
        seen.add(path)
        if prefetched_paths is not None and path in prefetched_paths:
            continue
        paths.append(path)
    return paths


def _prefetch_file(file_name: str, description: str) -> None:
    if not os.path.exists(file_name):
        return

    logger.info(f"Prefetching {description} file {file_name} to host page cache...")
    with open(file_name, "rb") as f:
        while f.read(_PREFETCH_CHUNK_SIZE):
            pass
    logger.info(f"Finished prefetching {description} file {file_name}.")


def prefetch_files_to_host_cache(
    file_names: Iterable[str],
    *,
    description: str,
    prefetched_paths: Optional[Set[str]] = None,
    ignore_errors: bool = False,
) -> bool:
    """Warm checkpoint files in host page cache across distributed local ranks.

    Returns True only when all selected files were already prefetched or were
    prefetched successfully. If prefetch is skipped or fails, returns False
    when ignore_errors=True and raises otherwise. The optional process-local
    cache records only files assigned to this rank.
    """
    local_paths: list[str] = []
    success = False
    try:
        # Cache hits and empty inputs must participate in the same collectives
        # as ranks that still need to read files.
        local_rank, local_size, available_memory = _local_prefetch_context()
        paths = _normalize_paths(file_names, None)
        # Keep shard ownership stable when process-local caches differ.
        local_paths = _normalize_paths(paths[local_rank::local_size], prefetched_paths)
        if not local_paths:
            success = True
            return True

        prefetch_size = sum(os.path.getsize(path) for path in paths if os.path.exists(path))
        if prefetch_size >= available_memory * 0.9:
            logger.info(
                f"Skipping {description} prefetch because files require "
                f"{prefetch_size / (1024**3):.2f}GB and available host memory is "
                f"{available_memory / (1024**3):.2f}GB."
            )
            return False

        if local_paths:
            logger.info(
                f"Prefetching {prefetch_size / (1024**3):.2f}GB {description} "
                "files across distributed local ranks."
            )
            max_workers = min(multiprocessing.cpu_count() * 2, 16, len(local_paths))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(lambda path: _prefetch_file(path, description), local_paths))

        success = True
        return True
    except Exception as exc:
        if not ignore_errors:
            raise
        logger.warning(f"{description} prefetch failed; continuing without prefetch: {exc}")
        return False
    finally:
        if success and prefetched_paths is not None:
            prefetched_paths.update(local_paths)
        _dist_barrier()
