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
"""Host-memory ``madvise`` helpers for releasing mmap-backed page cache.

These utilities advise the OS to drop the physical pages backing read-only
mmap regions (e.g. safetensors shards) so the resident file cache cannot grow
unbounded during weight load on host-memory-constrained nodes. They only emit
a kernel *hint*: the data is re-faulted on demand and is never corrupted.
"""

import ctypes
import mmap

__all__ = [
    "madvise_range",
    "pageout_file_backed_regions",
    "advise_tensor_pageout",
]

_MADV_DONTNEED = 4
_MADV_PAGEOUT = 21
_MADV_ADVICE_BY_MODE = {"dontneed": _MADV_DONTNEED, "pageout": _MADV_PAGEOUT}


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
        raise ValueError("mode must be 'pageout' or 'dontneed'.")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    ret = libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(size), ctypes.c_int(advice))
    if ret != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"madvise() failed with errno={err}")


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
    - Safe to call on mmap-backed tensors (data will be reloaded on next access).
    - If called on a malloc-based tensor (not mmap), madvise() simply does nothing
      and returns 0 (success) because the virtual address range is anonymous memory.
      It does NOT crash or corrupt data.
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
