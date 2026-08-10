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


def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, ""))
    except ValueError:
        return default


def _local_rank_and_size() -> tuple[int, int]:
    if not _dist_initialized():
        return 0, 1

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = _get_int_env("LOCAL_RANK", rank)
    local_size = _get_int_env("LOCAL_WORLD_SIZE", world_size)

    if local_size < 1 or local_rank < 0 or local_rank >= local_size:
        return rank, world_size
    return local_rank, local_size


def _get_local_available_host_memory() -> int:
    """Return the minimum available host memory observed by local ranks.

    The prefetch/skip decision must be the same for all ranks that synchronize
    at the post-prefetch barrier. Use torch.distributed to collect per-rank
    snapshots and reduce the local slice to its minimum.
    """
    available_memory = psutil.virtual_memory().available
    if not _dist_initialized() or dist.get_world_size() == 1:
        return available_memory

    world_size = dist.get_world_size()
    gathered_memory: list[int | None] = [None] * world_size
    dist.all_gather_object(gathered_memory, int(available_memory))

    local_rank, local_size = _local_rank_and_size()
    local_start = dist.get_rank() - local_rank
    local_end = min(local_start + local_size, world_size)
    local_memory = [
        memory for memory in gathered_memory[local_start:local_end] if memory is not None
    ]
    if not local_memory:
        return available_memory
    return min(local_memory)


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
    when ignore_errors=True and raises otherwise.
    """
    paths = _normalize_paths(file_names, prefetched_paths)
    success = False
    try:
        if not paths:
            success = True
            return True

        prefetch_size = sum(os.path.getsize(path) for path in paths if os.path.exists(path))
        available_memory = _get_local_available_host_memory()
        if prefetch_size >= available_memory * 0.9:
            logger.info(
                f"Skipping {description} prefetch because files require "
                f"{prefetch_size / (1024**3):.2f}GB and available host memory is "
                f"{available_memory / (1024**3):.2f}GB."
            )
            return False

        local_rank, local_size = _local_rank_and_size()
        local_paths = paths[local_rank::local_size]
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
            prefetched_paths.update(paths)
        _dist_barrier()
