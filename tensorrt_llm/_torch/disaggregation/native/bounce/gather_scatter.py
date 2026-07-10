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
"""Device-to-device gather and scatter for KV bounce. It coalesces the scattered fragments into one
contiguous buffer, and the inverse, in a single batched kernel launch, and falls back to a
per-fragment copy loop when Triton or the GPU is unavailable."""

import threading
from dataclasses import dataclass

import numpy as np

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.runtime.generation import CUASSERT

try:
    import torch
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except ImportError:  # keep importable without GPU or Triton at import time
    _HAVE_TRITON = False

_D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

# copy in 128-bit elements, the widest per-thread transaction, as two 64-bit words
_ELEM_BYTES = 16

# elements per program; an architecture-agnostic default of about 64 KiB of copy
_BLOCK = 4096
# fixed rather than autotuned, to avoid recompiling per shape on the hot path
_NUM_WARPS = 4
_CHUNK_BYTES = _BLOCK * _ELEM_BYTES


@dataclass
class Plan:
    """One transfer's coalescing plan: the source and destination fragment addresses, their sizes,
    and the total. Both sides walk the fragments in the same order, and the offsets are the running
    sum of the sizes."""

    src_ptrs: np.ndarray  # source fragment addresses
    dst_ptrs: np.ndarray  # destination fragment addresses
    sizes: np.ndarray  # bytes per fragment
    total_size: int  # total bytes; must not exceed the bounce buffer capacity

    @property
    def offsets(self) -> np.ndarray:
        if self.sizes.size == 0:
            return np.zeros(0, dtype=np.int64)
        off = np.empty(self.sizes.size, dtype=np.int64)
        off[0] = 0
        np.cumsum(self.sizes[:-1], out=off[1:])
        return off


if _HAVE_TRITON:

    @triton.jit
    def _batched_copy_kernel(
        dst_ptrs_ptr,  # destination address per fragment
        src_ptrs_ptr,  # source address per fragment
        nvec_ptr,  # 128-bit element count per fragment
        n_frags,  # number of fragments
        BLOCK: tl.constexpr,  # elements copied per program
    ):
        # each program copies one block-sized slice of one fragment
        frag = tl.program_id(0)
        chunk = tl.program_id(1)
        if frag >= n_frags:
            return

        dst_addr = tl.load(dst_ptrs_ptr + frag)
        src_addr = tl.load(src_ptrs_ptr + frag)
        nvec = tl.load(nvec_ptr + frag)

        vec = chunk * BLOCK + tl.arange(0, BLOCK)  # element index within the fragment
        mask = vec < nvec  # masks off the fragment's tail

        # each row is two 64-bit words the compiler coalesces into one 128-bit access
        i64 = vec[:, None] * 2 + tl.arange(0, 2)[None, :]
        m2 = mask[:, None]
        src_i64 = src_addr.to(tl.uint64).to(tl.pointer_type(tl.int64))
        dst_i64 = dst_addr.to(tl.uint64).to(tl.pointer_type(tl.int64))

        vals = tl.load(src_i64 + i64, mask=m2)
        tl.store(dst_i64 + i64, vals, mask=m2)


def _uniform_nelem(sizes: np.ndarray):
    """Return the element counts, the maximum, and whether every fragment is 16-byte aligned, which
    the 128-bit copy requires."""
    if sizes.size == 0:
        return None, 0, False
    if not np.all((sizes % _ELEM_BYTES) == 0):
        return None, 0, False
    nvec = (sizes // _ELEM_BYTES).astype(np.int64)
    return nvec, int(nvec.max()), True


# reusable pinned staging for the metadata copy, one per stream, so the copy is a true async transfer
_meta_lock = threading.Lock()
_meta_buffers = {}  # each stream maps to its pinned host buffer, device buffer, and capacity


def _get_meta_buffers(stream_handle: int, need: int, dev):
    """Return the pinned host and device buffers for the stream, large enough for the request,
    growing them under a lock."""
    with _meta_lock:
        ent = _meta_buffers.get(stream_handle)
        if ent is None or ent[2] < need:
            new_cap = max(need, (ent[2] * 2 if ent else 0), 4096)
            pinned = torch.empty(new_cap, dtype=torch.int64, pin_memory=prefer_pinned())
            devt = torch.empty(new_cap, dtype=torch.int64, device=dev)
            ent = (pinned, devt, new_cap)
            _meta_buffers[stream_handle] = ent
        return ent[0], ent[1]


def _launch_batched_copy(
    dst_addrs: np.ndarray, src_addrs: np.ndarray, sizes: np.ndarray, stream
) -> bool:
    """Run the single batched copy on the stream. Returns False when the caller must use the loop
    fallback."""
    if not _HAVE_TRITON or not torch.cuda.is_available():
        return False

    n = int(src_addrs.size)
    if n == 0:
        return True  # nothing to copy, trivially done

    nvec, max_nvec, ok = _uniform_nelem(sizes)
    if not ok or max_nvec == 0:
        return False

    dev = torch.device("cuda", torch.cuda.current_device())
    # pack all three address arrays into one pinned buffer and one async copy, so the kernel is
    # ordered after it
    stream_handle = int(stream)
    pinned, devt = _get_meta_buffers(stream_handle, 3 * n, dev)
    host = pinned.numpy()
    host[:n] = dst_addrs
    host[n : 2 * n] = src_addrs
    host[2 * n : 3 * n] = nvec

    n_chunks = triton.cdiv(max_nvec, _BLOCK)
    grid = (n, n_chunks)

    # buffer reuse is safe: every caller syncs the stream between calls, so the previous copy and
    # kernel finish before the refill
    ext_stream = torch.cuda.ExternalStream(stream_handle)
    with torch.cuda.stream(ext_stream):
        devt[: 3 * n].copy_(pinned[: 3 * n], non_blocking=True)
        _batched_copy_kernel[grid](
            devt[:n],
            devt[n : 2 * n],
            devt[2 * n : 3 * n],
            n,
            BLOCK=_BLOCK,
            num_warps=_NUM_WARPS,
        )
    return True


def _copy_frags(pairs, sizes: np.ndarray, stream) -> None:
    """Fallback used when the batched copy is unavailable: one async copy per fragment."""
    # strict zip: a length mismatch must fail fast rather than silently drop part of the copy
    for (dst, src), n in zip(pairs, sizes, strict=True):
        CUASSERT(cudart.cudaMemcpyAsync(int(dst), int(src), int(n), _D2D, stream))


def gather_contiguous(
    dst_base: int,
    src_ptrs: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    *,
    stream,
) -> None:
    """Gather each source fragment into its place in the contiguous buffer, asynchronously. The
    caller syncs before issuing the write."""
    src_addrs = np.asarray(src_ptrs, dtype=np.int64)
    dst_addrs = np.int64(dst_base) + np.asarray(offsets, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)

    if _launch_batched_copy(dst_addrs, src_addrs, sizes, stream):
        return

    _copy_frags(
        ((int(dst_addrs[k]), int(src_addrs[k])) for k in range(src_addrs.size)),
        sizes,
        stream,
    )


def scatter_contiguous(
    src_base: int,
    dst_ptrs: np.ndarray,
    sizes: np.ndarray,
    offsets: np.ndarray,
    *,
    stream,
) -> None:
    """The inverse of gather: scatter each piece of the contiguous buffer back to its destination
    fragment, asynchronously. The caller syncs before signaling completion."""
    dst_addrs = np.asarray(dst_ptrs, dtype=np.int64)
    src_addrs = np.int64(src_base) + np.asarray(offsets, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)

    if _launch_batched_copy(dst_addrs, src_addrs, sizes, stream):
        return

    _copy_frags(
        ((int(dst_addrs[k]), int(src_addrs[k])) for k in range(dst_addrs.size)),
        sizes,
        stream,
    )
