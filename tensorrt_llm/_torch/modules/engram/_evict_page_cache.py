# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evict one host NUMA node's clean page cache by faulting in anonymous memory bound to the node.

usage: python _evict_page_cache.py <node> <chunk bytes> <margin bytes>
"""

import ctypes
import mmap
import os
import platform
import sys

_MPOL_BIND = 2
_MADV_POPULATE_WRITE = 23
_MBIND_SYSCALL = {"aarch64": 235, "x86_64": 237}


def _meminfo(node: int) -> dict:
    """``node``'s meminfo fields, with kB values in bytes."""
    fields = {}
    with open(f"/sys/devices/system/node/node{node}/meminfo") as meminfo:
        for line in meminfo:
            parts = line.split()
            fields[parts[2].rstrip(":")] = int(parts[3]) * 1024
    return fields


def _reclaimable_cache(node: int) -> int:
    """Clean file pages on the LRU lists of ``node``: what reclaim can drop without writeback."""
    fields = _meminfo(node)
    return fields["Active(file)"] + fields["Inactive(file)"] - fields["Dirty"] - fields["Writeback"]


def _cgroup_headroom() -> float:
    """Memory this process's cgroup (v2) and its visible ancestors can still charge before a limit."""
    headroom = float("inf")
    try:
        with open("/proc/self/cgroup") as cgroup:
            path = next(line.split(":", 2)[2].strip() for line in cgroup if line.startswith("0::"))
    except (OSError, StopIteration):
        return headroom
    directory = os.path.normpath("/sys/fs/cgroup" + path)
    # A cgroup outside this namespace (``0::/../...``) normalizes out of /sys/fs/cgroup and shows no limit.
    while directory == "/sys/fs/cgroup" or directory.startswith("/sys/fs/cgroup/"):
        try:
            with open(os.path.join(directory, "memory.max")) as limit:
                text = limit.read().strip()
            if text != "max":
                with open(os.path.join(directory, "memory.current")) as current:
                    headroom = min(headroom, int(text) - int(current.read()))
        except (OSError, ValueError):
            pass
        directory = os.path.dirname(directory)
    return headroom


def main() -> int:
    node, chunk, margin = (int(arg) for arg in sys.argv[1:4])
    # The margin keeps the node from running out; if it still does, the OOM killer picks this process first.
    with open("/proc/self/oom_score_adj", "w") as score:
        score.write("1000")
    # Room for the whole node, so cache the workers refill during the eviction goes too.
    size = _meminfo(node)["MemTotal"] // chunk * chunk
    mapping = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    address = ctypes.addressof(ctypes.c_uint8.from_buffer(mapping))
    mask = ctypes.c_ulong(1 << node)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    if libc.syscall(
        ctypes.c_long(_MBIND_SYSCALL[platform.machine()]),
        ctypes.c_void_p(address),
        ctypes.c_ulong(size),
        ctypes.c_int(_MPOL_BIND),
        ctypes.byref(mask),
        ctypes.c_ulong(64),
        ctypes.c_uint(0),
    ):
        return 1
    # Each step only takes memory the node can still free from its page cache, within any cgroup limit.
    for offset in range(0, size, chunk):
        if min(_reclaimable_cache(node), _cgroup_headroom()) < chunk + margin:
            break
        mapping.madvise(_MADV_POPULATE_WRITE, offset, chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
