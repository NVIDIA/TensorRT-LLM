# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path


def _read_meminfo_available_bytes(meminfo_path: str = "/proc/meminfo") -> int | None:
    """Read the kernel's MemAvailable estimate in bytes from a procfs file."""
    try:
        for line in Path(meminfo_path).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return max(0, int(line.split()[1]) * 1024)
    except (OSError, UnicodeError, ValueError, IndexError):
        return None
    return None


def _read_cgroup_headroom(cgroup: Path) -> int | None:
    """Read one cgroup v2 limit, crediting only readily reclaimable file cache."""
    try:
        limit_text = (cgroup / "memory.max").read_text().strip()
        if limit_text == "max":
            return None
        limit = int(limit_text)
        current = int((cgroup / "memory.current").read_text())
    except (OSError, UnicodeError, ValueError):
        return None

    reclaimable = 0
    try:
        stats = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
        # Exclude shmem, pages awaiting writeback, and unevictable pages.
        # These counters can overlap, and unevictable includes anonymous
        # pages, so subtracting all is deliberately conservative. Missing
        # counters get no file-cache credit.
        unavailable = sum(
            max(0, int(stats[key]))
            for key in ("shmem", "file_dirty", "file_writeback", "unevictable")
        )
        reclaimable = max(0, int(stats["file"]) - unavailable)
    except (OSError, UnicodeError, ValueError, KeyError):
        pass
    return max(0, limit - max(0, current - reclaimable))


def _read_cgroup_memory_headroom_bytes(
    cgroup_dir: str = "/sys/fs/cgroup", proc_cgroup_path: str = "/proc/self/cgroup"
) -> int | None:
    """Return the tightest finite headroom from the process cgroup and ancestors.

    Args:
        cgroup_dir: Root of the cgroup v2 mount.
        proc_cgroup_path: Procfs file identifying the process's cgroup.
    """
    root = Path(cgroup_dir)
    leaf = root
    try:
        for line in Path(proc_cgroup_path).read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and not controllers:
                parts = Path(relative.lstrip("/")).parts
                if ".." not in parts:
                    candidate = root.joinpath(*parts)
                    if candidate.is_dir():
                        leaf = candidate
                break
    except (OSError, UnicodeError, ValueError):
        pass

    # A private namespace or a subtree mount can expose the process cgroup at
    # the mount root, even if the reported host-relative path is not visible.
    headrooms = []
    while True:
        headroom = _read_cgroup_headroom(leaf)
        if headroom is not None:
            headrooms.append(headroom)
        if leaf == root:
            break
        leaf = leaf.parent
    return min(headrooms) if headrooms else None


def get_integrated_gpu_free_memory(cuda_free: int, cuda_total: int) -> int:
    """Estimate allocatable shared memory, including reclaimable host page cache.

    MemAvailable can include file cache that CUDA does not report as free on
    integrated GPUs. Finite cgroup headroom is a hard upper bound, including
    when it is below CUDA free or MemAvailable cannot be read.

    Args:
        cuda_free: Free bytes reported by CUDA; used if MemAvailable is absent.
        cuda_total: Device capacity in bytes, an upper bound on the estimate.
    """
    available = _read_meminfo_available_bytes()
    available = cuda_free if available is None else max(cuda_free, available)
    available = min(available, cuda_total)
    cgroup_headroom = _read_cgroup_memory_headroom_bytes()
    if cgroup_headroom is not None:
        available = min(available, cgroup_headroom)
    return max(0, available)
