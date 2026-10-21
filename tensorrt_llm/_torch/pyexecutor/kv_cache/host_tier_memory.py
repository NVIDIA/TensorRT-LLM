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
"""Opt-in host-memory preparation around the KV cache manager v2 host tier.

The host tier is anonymous memory advised for transparent huge pages. Where the
huge-page faults fall back to base pages when the tier is allocated, the kernel
later collapses those ranges (khugepaged) and migrates their pages (compaction)
while the server runs, and whatever touches a range or page in transit waits.
These helpers move that work to startup. Each is a no-op unless its
environment variable is set to ``1``:

* ``TRTLLM_KV_HOST_TIER_DROP_PAGE_CACHE``: before the KV cache managers are
  built, drop this node's page cache of the checkpoint files with
  ``posix_fadvise(POSIX_FADV_DONTNEED)``. The host tier is then allocated from
  memory the cache no longer holds, and the cache is not released later, while
  serving.
* ``TRTLLM_KV_HOST_TIER_COLLAPSE``: after the KV cache manager v2 has created
  its host tier, collapse the new host-tier mappings into huge pages with
  ``madvise(MADV_COLLAPSE)``.

Neither needs privileges. Failures are logged and otherwise ignored.
"""

import ctypes
import os
import time
from typing import Iterable, List, Optional, Tuple

from tensorrt_llm.logger import logger

DROP_PAGE_CACHE_ENV = "TRTLLM_KV_HOST_TIER_DROP_PAGE_CACHE"
COLLAPSE_ENV = "TRTLLM_KV_HOST_TIER_COLLAPSE"

# From <linux/mman.h>; Python's mmap module does not export it.
_MADV_COLLAPSE = 25
_MIN_MAPPING_BYTES = 64 << 20

AddressRange = Tuple[int, int]


def drop_page_cache_enabled() -> bool:
    return os.environ.get(DROP_PAGE_CACHE_ENV) == "1"


def collapse_enabled() -> bool:
    return os.environ.get(COLLAPSE_ENV) == "1"


def _iter_files(roots: Iterable[Optional[str]]) -> Iterable[str]:
    seen = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                path = os.path.realpath(os.path.join(dirpath, name))
                if path not in seen and os.path.isfile(path):
                    seen.add(path)
                    yield path


def drop_page_cache(roots: Iterable[Optional[str]]) -> None:
    """Drop this node's page cache of every regular file under ``roots``."""
    start = time.monotonic()
    num_files = 0
    num_bytes = 0
    for path in _iter_files(roots):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            num_bytes += os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            num_files += 1
        except OSError as error:
            logger.warning(f"{DROP_PAGE_CACHE_ENV}: posix_fadvise({path}) failed: {error}")
        finally:
            os.close(fd)
    logger.info(
        f"{DROP_PAGE_CACHE_ENV}: dropped the page cache of {num_files} files "
        f"({num_bytes / (1 << 30):.1f} GiB) in {time.monotonic() - start:.1f} s"
    )


def anonymous_mappings() -> List[AddressRange]:
    """Return the private anonymous read-write mappings of at least 64 MiB."""
    ranges = []
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split()
                # An anonymous mapping has no pathname field.
                if len(fields) != 5 or fields[1] != "rw-p":
                    continue
                begin, end = (int(x, 16) for x in fields[0].split("-"))
                if end - begin >= _MIN_MAPPING_BYTES:
                    ranges.append((begin, end))
    except OSError as error:
        logger.warning(f"{COLLAPSE_ENV}: cannot read /proc/self/maps: {error}")
    return ranges


def _anon_huge_bytes() -> int:
    try:
        with open("/proc/self/smaps_rollup") as rollup:
            for line in rollup:
                if line.startswith("AnonHugePages:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return -1


def _huge_page_bytes() -> int:
    try:
        with open("/sys/kernel/mm/transparent_hugepage/hpage_pmd_size") as f:
            return int(f.read())
    except (OSError, ValueError):
        return 0


def collapse_new_mappings(before: List[AddressRange]) -> None:
    """``MADV_COLLAPSE`` the anonymous mappings that are not in ``before``."""
    huge = _huge_page_bytes()
    known = set(before)
    targets = [r for r in anonymous_mappings() if r not in known]
    if huge <= 0 or not targets:
        logger.info(
            f"{COLLAPSE_ENV}: nothing to collapse (huge page {huge} B, {len(targets)} new mappings)"
        )
        return
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.madvise.restype = ctypes.c_int
    start = time.monotonic()
    huge_before = _anon_huge_bytes()
    collapsed = 0
    errors = {}
    for begin, end in targets:
        first = (begin + huge - 1) // huge * huge
        for addr in range(first, end - huge + 1, huge):
            if libc.madvise(ctypes.c_void_p(addr), huge, _MADV_COLLAPSE) == 0:
                collapsed += 1
            else:
                err = ctypes.get_errno()
                errors[err] = errors.get(err, 0) + 1
    total = sum(end - begin for begin, end in targets)
    logger.info(
        f"{COLLAPSE_ENV}: {len(targets)} new mappings ({total / (1 << 30):.1f} GiB): "
        f"{collapsed} huge pages collapsed or already huge, failures by errno {errors}; "
        f"AnonHugePages {huge_before / (1 << 30):.1f} -> {_anon_huge_bytes() / (1 << 30):.1f} GiB "
        f"in {time.monotonic() - start:.1f} s"
    )
