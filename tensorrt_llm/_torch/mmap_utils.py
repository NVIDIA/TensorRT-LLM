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
"""Host-memory `mmap`/`madvise`/`fadvise` helpers for file-backed page cache.

These utilities advise the OS to drop the physical pages backing read-only
file-backed mmap regions (e.g. safetensors shards) so the resident file cache
cannot grow unbounded during weight load on host-memory-constrained nodes,
fault a file's pages into the page cache ahead of use (`populate_file_pages`),
or drop a file's page cache once its data is no longer needed
(`release_file_page_cache`) and wait for the page migration that the freed
memory triggers (`wait_for_page_migration_to_settle`). Callers must establish
that a tensor is file-backed before using `advise_tensor_pageout`.
"""

import ctypes
import errno as errno_codes
import mmap
import os
import time
from collections.abc import Callable, Iterable

__all__ = [
    "madvise_range",
    "populate_file_pages",
    "pageout_file_backed_regions",
    "advise_tensor_pageout",
    "release_file_page_cache",
    "page_cache_bytes",
    "wait_for_page_migration_to_settle",
]

_MADV_DONTNEED = 4
_MADV_PAGEOUT = 21
_MADV_POPULATE_READ = 22  # Requires Linux >= 5.14.
_MADV_POPULATE_WRITE = 23  # Requires Linux >= 5.14.
_MADV_ADVICE_BY_MODE = {
    "dontneed": _MADV_DONTNEED,
    "pageout": _MADV_PAGEOUT,
    "populate_write": _MADV_POPULATE_WRITE,
}
_MMAP_FAILED = ctypes.c_void_p(-1).value


def madvise_range(addr: int, size: int, mode: str = "dontneed") -> None:
    """Issue ``madvise(addr, size, advice)`` over a page-aligned address range.

    Low-level shared wrapper around the ``libc.madvise`` syscall. ``addr`` and
    ``size`` must already be page-aligned -- both mmap regions (from
    ``/proc/self/maps``) and the clipped tensor ranges computed by
    ``advise_tensor_pageout`` satisfy this.

    Parameters
    ----------
    addr : int
        Start address of the range.
    size : int
        Length of the range in bytes. A non-positive size is a no-op.
    mode : str, optional
        "dontneed" -> MADV_DONTNEED (immediate discard, default)
        "pageout"  -> MADV_PAGEOUT  (asynchronous pageout, Linux 4.5+)
        "populate_write" -> MADV_POPULATE_WRITE (fault every page in writable, Linux 5.14+)

    Raises
    ------
    ValueError
        If an invalid mode is given.
    OSError
        If the madvise() syscall fails (errno will be included).
    """
    if size <= 0:
        return
    try:
        advice = _MADV_ADVICE_BY_MODE[mode]
    except KeyError:
        raise ValueError("mode must be 'pageout', 'dontneed' or 'populate_write'.")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    ret = libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(size), ctypes.c_int(advice))
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"madvise() failed with errno={err}")


def _log_populate_stop(message: str) -> None:
    # Lazy import keeps this module importable standalone (no torch); the
    # debug line carries the errno so a stopped populate is diagnosable.
    try:
        from tensorrt_llm.logger import logger
    except ImportError:
        return
    logger.debug(message)


def _errno_name(err: int) -> str:
    return errno_codes.errorcode.get(err, "unknown")


def populate_file_pages(
    file_name: str, window_bytes: int, on_window: Callable[[int], None] | None = None
) -> int:
    """Fault a file's pages into the OS page cache without copying to user space.

    Maps the file read-only and issues `madvise(MADV_POPULATE_READ)` over it in
    `window_bytes` windows. Windowing keeps per-call `mmap_lock` hold times
    short, so concurrent `mmap`/`munmap` callers in the process are not stalled
    behind one long populate, and gives callers progress granularity via
    `on_window(num_bytes)`. `window_bytes` must be a positive multiple of the
    page size (raises `ValueError` otherwise): a zero window would never advance,
    and an unaligned one would fail from the second window on with an error that is
    indistinguishable from an unsupported kernel.

    Both the mapping and the advice are issued through `ctypes` rather than the
    built-in `mmap` module deliberately. Population blocks for the duration of
    the underlying file read, and `mmap.mmap.madvise` holds the GIL for the whole
    call while `ctypes` releases it, so the built-in method would serialize
    concurrent populating threads (measured on a cold file: a background thread ran
    ~500x slower during an `mmap.madvise` populate than during the `ctypes`
    equivalent). The GIL-free `madvise` in turn needs the mapping's base address,
    which is why the mapping also comes from `libc`: a read-only `mmap.mmap`
    never exposes its address (`ctypes.*.from_buffer` requires a writable
    buffer), and mapping writable/private just to extract one would reintroduce
    the class of copy-on-write anonymous pages this populate exists to avoid.

    Returns the number of bytes populated. `MADV_POPULATE_READ` requires
    Linux >= 5.14 and an mmap-capable filesystem; when unsupported (or on any other
    failure) population stops early -- typically returning 0 -- and the caller is
    expected to warm the remaining bytes by other means.
    """
    if window_bytes <= 0 or window_bytes % mmap.PAGESIZE != 0:
        raise ValueError("window_bytes must be a positive multiple of the page size.")
    try:
        fd = os.open(file_name, os.O_RDONLY)
    except OSError as e:
        _log_populate_stop(f"populate_file_pages: open('{file_name}') failed: {e}")
        return 0
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return 0
        # ctypes rather than mmap.mmap: see the GIL / base-address note in the
        # docstring.
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.mmap.restype = ctypes.c_void_p  # Default int restype truncates on 64-bit.
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, _MMAP_FAILED):
            err = ctypes.get_errno()
            _log_populate_stop(
                f"populate_file_pages: mmap('{file_name}') failed with errno {err} "
                f"({_errno_name(err)})"
            )
            return 0
    except OSError as e:
        # Same contract as an unsupported kernel/filesystem: stop early and
        # let the caller warm the file by other means.
        _log_populate_stop(f"populate_file_pages: setup failed for '{file_name}': {e}")
        return 0
    finally:
        os.close(fd)  # The mapping keeps its own reference to the file.
    populated = 0
    try:
        while populated < size:
            length = min(window_bytes, size - populated)
            ret = libc.madvise(
                ctypes.c_void_p(addr + populated),
                ctypes.c_size_t(length),
                ctypes.c_int(_MADV_POPULATE_READ),
            )
            if ret != 0:
                err = ctypes.get_errno()
                _log_populate_stop(
                    f"populate_file_pages: madvise(MADV_POPULATE_READ) failed at offset "
                    f"{populated} of '{file_name}' with errno {err} ({_errno_name(err)})"
                )
                break
            populated += length
            if on_window is not None:
                on_window(length)
    finally:
        libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(size))
    return populated


def pageout_file_backed_regions(path_substring: str, mode: str = "dontneed") -> None:
    """``madvise`` every mmap region whose backing file path matches a substring.

    Scans ``/proc/self/maps`` and advises ``MADV_DONTNEED`` / ``MADV_PAGEOUT``
    on each mapped region whose path contains ``path_substring``. Used to bound
    the resident file-cache of large read-only mmaps (e.g. the safetensors
    shards) during weight load on host-memory-constrained nodes. mmap regions
    are always page-aligned, so the raw ``[start, end)`` bounds parsed from the
    maps file are passed straight to ``madvise_range``. Best-effort: per-region
    failures are swallowed so a transient unmap cannot abort the caller.
    """
    try:
        maps = open("/proc/self/maps")
    except OSError:
        return
    with maps:
        for line in maps:
            if path_substring not in line:
                continue
            start_hex, end_hex = line.split()[0].split("-")
            start = int(start_hex, 16)
            try:
                madvise_range(start, int(end_hex, 16) - start, mode)
            except OSError:
                pass


def advise_tensor_pageout(tensor, mode: str = "dontneed"):
    """
    Advise the OS to page out or discard the physical memory pages backing a CPU tensor.
    This works only for tensors backed by an mmap'ed file or shared memory.

    Parameters
    ----------
    tensor : torch.Tensor
        A CPU tensor (usually created via torch.from_file() or numpy.memmap()).
    mode : str, optional
        "pageout"  -> use MADV_PAGEOUT (asynchronous pageout, Linux 4.5+)
        "dontneed" -> use MADV_DONTNEED (immediate discard)

    Raises
    ------
    ValueError
        If the tensor is not on CPU or an invalid mode is given.
    OSError
        If the madvise() syscall fails (errno will be included).

    Notes
    -----
    - Works only on Linux systems.
    - This call only gives a *hint* to the kernel: the OS may decide to ignore it.
    - Safe to call on file-backed mmap tensors (data will be reloaded on next
      access).
    - Do not call this on malloc-based tensors. ``MADV_DONTNEED`` may discard
      anonymous pages, and later accesses can observe zero-filled memory.
    """

    if not tensor.device.type == "cpu":
        raise ValueError("Only CPU tensors are supported.")

    # Get raw pointer and size in bytes
    ptr = tensor.data_ptr()
    nbytes = tensor.numel() * tensor.element_size()

    # Only operate on complete pages within the tensor's memory range
    # to avoid affecting memory outside the tensor boundaries
    page_size = mmap.PAGESIZE

    # Round up to the first complete page boundary inside the tensor
    start_aligned = (ptr + page_size - 1) & ~(page_size - 1)

    # Round down to the last complete page boundary inside the tensor
    end_aligned = (ptr + nbytes) & ~(page_size - 1)

    # madvise only the complete pages fully inside the tensor's bounds.
    madvise_range(start_aligned, end_aligned - start_aligned, mode)


_MEMINFO_PATH = "/proc/meminfo"
_VMSTAT_PATH = "/proc/vmstat"


def release_file_page_cache(paths: Iterable[str]) -> int:
    """Advise the kernel to drop the page cache of ``paths`` (``POSIX_FADV_DONTNEED``).

    Returns the total size in bytes of the files advised, not the bytes
    dropped: the kernel keeps pages that are dirty or mapped by a process.
    Files that cannot be opened or advised are skipped.
    """
    advised_bytes = 0
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            advised_bytes += os.fstat(fd).st_size
        except OSError:
            pass
        finally:
            os.close(fd)
    return advised_bytes


def page_cache_bytes() -> int | None:
    """Bytes of file data in the host's page cache (``Cached`` in ``/proc/meminfo``).

    Returns None if it cannot be read.
    """
    try:
        with open(_MEMINFO_PATH) as meminfo:
            for line in meminfo:
                if line.startswith("Cached:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _migrated_pages() -> int | None:
    try:
        with open(_VMSTAT_PATH) as vmstat:
            for line in vmstat:
                if line.startswith("pgmigrate_success "):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def wait_for_page_migration_to_settle(
    max_wait_s: float = 120.0,
    quiet_s: float = 5.0,
    poll_s: float = 0.5,
    max_bytes_per_s: float = 128 * 2**20,
) -> float:
    """Wait until the kernel has stopped migrating pages, or ``max_wait_s`` passes.

    Freeing a large page cache at once leaves free memory fragmented, and the
    kernel compacts it by migrating in-use pages. A thread that touches a page
    under migration blocks until the copy finishes. Polls ``pgmigrate_success``
    in ``/proc/vmstat`` until the migration rate, in bytes per second over the
    measured interval between reads, stays at or below ``max_bytes_per_s`` for
    ``quiet_s`` seconds.

    Returns the seconds waited: 0 if ``pgmigrate_success`` cannot be read at
    the start, and the time waited so far if a later read fails.
    """
    start = time.monotonic()
    previous = _migrated_pages()
    if previous is None:
        return 0.0
    previous_time = quiet_since = start
    while True:
        time.sleep(poll_s)
        now = time.monotonic()
        current = _migrated_pages()
        if current is None:
            return now - start
        interval = now - previous_time
        if interval > 0 and ((current - previous) * mmap.PAGESIZE / interval > max_bytes_per_s):
            quiet_since = now
        previous, previous_time = current, now
        if now - quiet_since >= quiet_s or now - start >= max_wait_s:
            return now - start
