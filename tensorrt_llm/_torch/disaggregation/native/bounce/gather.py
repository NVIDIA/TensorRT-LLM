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
"""D2D gather/scatter for KV bounce: coalesce N scattered fragments into one contiguous
buffer (and the inverse) via a SINGLE batched Triton launch, falling back to a per-fragment
copy loop when Triton/GPU is absent."""

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
except ImportError:  # keep importable without GPU / triton at import time
    _HAVE_TRITON = False

_D2D = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

# Copy in 128-bit elements (the widest per-thread transaction) as 2 contiguous int64 since Triton has no uint128.
_ELEM_BYTES = 16

# 64 KiB/program; arch-agnostic default across uniform/small/non-uniform frags.
_BLOCK = 4096
# Fixed (not autotuned) to avoid per-shape recompilation on the hot path.
_NUM_WARPS = 4
_CHUNK_BYTES = _BLOCK * _ELEM_BYTES


@dataclass
class Plan:
    """One transfer's coalescing plan: gather src_ptrs[k]->buf[offsets[k]] and the
    inverse scatter, iterated in the SAME order both sides (offsets = cumsum of sizes)."""

    src_ptrs: np.ndarray  # int64[N] scattered SOURCE fragment addrs (sender KV pool)
    dst_ptrs: np.ndarray  # int64[N] scattered DEST fragment addrs (receiver KV pool)
    sizes: np.ndarray  # int64[N] per-fragment bytes (uniform per pool)
    total_size: int  # == int(sizes.sum()); MUST be <= bounce buffer capacity

    @property
    def num_frags(self) -> int:
        return int(self.sizes.size)

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
        dst_ptrs_ptr,  # int64[N] absolute dst byte address per fragment
        src_ptrs_ptr,  # int64[N] absolute src byte address per fragment
        nvec_ptr,  # int64[N] 16-byte (128-bit) element count per fragment
        n_frags,  # number of fragments
        BLOCK: tl.constexpr,  # 16-byte elements copied per program
    ):
        # program (frag, chunk) copies one BLOCK-element slice of one fragment.
        frag = tl.program_id(0)
        chunk = tl.program_id(1)
        if frag >= n_frags:
            return

        dst_addr = tl.load(dst_ptrs_ptr + frag)
        src_addr = tl.load(src_ptrs_ptr + frag)
        nvec = tl.load(nvec_ptr + frag)

        vec = chunk * BLOCK + tl.arange(0, BLOCK)  # 16-byte element index
        mask = vec < nvec  # masks the fragment's tail chunk

        # [BLOCK,2] int64 tile -> compiler coalesces each row into one 128-bit access.
        i64 = vec[:, None] * 2 + tl.arange(0, 2)[None, :]
        m2 = mask[:, None]
        src_i64 = src_addr.to(tl.uint64).to(tl.pointer_type(tl.int64))
        dst_i64 = dst_addr.to(tl.uint64).to(tl.pointer_type(tl.int64))

        vals = tl.load(src_i64 + i64, mask=m2)
        tl.store(dst_i64 + i64, vals, mask=m2)


def _uniform_nelem(sizes: np.ndarray):
    """Return (nvec_int64, max_nvec, ok); ok is False unless every size is 16B-aligned (128-bit copy precondition)."""
    if sizes.size == 0:
        return None, 0, False
    if not np.all((sizes % _ELEM_BYTES) == 0):
        return None, 0, False
    nvec = (sizes // _ELEM_BYTES).astype(np.int64)
    return nvec, int(nvec.max()), True


# Reusable pinned metadata-H2D staging, keyed per stream, so the H2D is a true async DMA.
_meta_lock = threading.Lock()
_meta_buffers = {}  # stream_handle -> (pinned_host[int64], device[int64], capacity)


def _get_meta_buffers(stream_handle: int, need: int, dev):
    """Return (pinned_host, device) tensors with >= need int64 capacity for the stream, growing under a lock."""
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
    """Run the single batched Triton copy (absolute per-fragment addrs) on stream; False => caller uses the loop
    fallback."""
    if not _HAVE_TRITON or not torch.cuda.is_available():
        return False

    n = int(src_addrs.size)
    if n == 0:
        return True  # nothing to copy; trivially done

    nvec, max_nvec, ok = _uniform_nelem(sizes)
    if not ok or max_nvec == 0:
        return False

    dev = torch.device("cuda", torch.cuda.current_device())
    # Pack dst|src|nvec into ONE pinned buffer + ONE async H2D on the copy stream so the kernel is ordered after it.
    stream_handle = int(stream)
    pinned, devt = _get_meta_buffers(stream_handle, 3 * n, dev)
    host = pinned.numpy()
    host[:n] = dst_addrs
    host[n : 2 * n] = src_addrs
    host[2 * n : 3 * n] = nvec

    n_chunks = triton.cdiv(max_nvec, _BLOCK)
    grid = (n, n_chunks)

    # Buffer reuse is safe: every caller syncs the stream between calls, so the prior H2D+kernel finish before refill.
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
    """Fallback (no Triton/GPU or unaligned): one async D2D copy per fragment; pairs yields (dst_ptr, src_ptr)."""
    # strict=True: a pairs/sizes length mismatch must fail fast, not silently drop part of the KV copy.
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
    """Gather each fragment src_ptrs[k] -> dst_base+offsets[k] (contiguous), async on stream; caller syncs before the
    WRITE."""
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
    """Inverse of gather: scatter src_base+offsets[k] -> dst_ptrs[k], async on stream; caller syncs before signaling
    completion."""
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
