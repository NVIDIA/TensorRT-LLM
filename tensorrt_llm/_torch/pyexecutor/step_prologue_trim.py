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
"""Step-prologue trim (``TRTLLM_STEP_PROLOGUE_TRIM``; unset / "1" = on, "0" = the
original path, "verify" = on plus a self-check at every elided write).

Under the overlap scheduler the host prepares step N+1 while step N's decode
CUDA graph is still running, so every eager op of the step-N+1 prologue is
already queued when graph N ends. The step boundary is then bounded by the GPU
draining that serial chain of tiny ops (H2D copies from pinned memory count
the same as kernels), not by host time. Host work is effectively free at this
point, so the trim moves work from the stream to the host:

* ``UnchangedCopyFilter``: skip an H2D / table copy whose host source is
  bit-identical to what this same code path last wrote to the same device
  region (block tables change only when a block is appended; slot indices,
  gather ids and prompt lengths are step-invariant in steady decode).
* Dead-store elimination for DSA decode metadata that the forward's
  ``on_update_kv_lens()`` (run at the top of every ``_forward_step``, inside
  the decode graph under replay) rebuilds before any reader.
* Fewer launches for the overlap-scheduler input fix-up (``index_select`` with
  int32 indices straight into the destination instead of index cast +
  advanced-indexing gather + D2D copy).
* One-model speculative sampling on the execution stream (no cross-stream
  event hops at the step boundary).

All changes are numerics-preserving: the device buffers hold exactly the
values the original path produces whenever they are read. ``verify`` mode
checks that on a live run: every copy the filter / the engine would elide is
issued (or its expected content computed) anyway and compared with what the
device held before, raising ``StepPrologueTrimVerifyError`` on any difference.

The mode is read from the environment once, at import (``reload_mode()``
re-reads it for tests): the gate is consulted some 15-30 times per step, and a
mid-run flip must not land inside a captured region (the same rationale as
``_fused_dsa_meta_enabled()`` in ``dsa/metadata.py``).
"""

import os
from typing import Any, Dict, Hashable, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

_ENV = "TRTLLM_STEP_PROLOGUE_TRIM"

# Largest host payload the filter snapshots and compares. The compare is a
# host memcmp that is off the critical path while the host runs a step ahead
# of the GPU (small batches), but the tables grow with the batch while the
# odds that no row changed this step shrink (any request crossing a block
# boundary rewrites the table), so fall back to plain copies there.
# 512 KiB = eight 16k-block int32 rows.
MAX_COMPARE_BYTES = 512 * 1024

_MODE = "1"
_ENABLED = True
_VERIFY = False


def reload_mode() -> str:
    """Re-read ``TRTLLM_STEP_PROLOGUE_TRIM`` into the module constants (tests)."""
    global _MODE, _ENABLED, _VERIFY
    _MODE = os.environ.get(_ENV, "1")
    _ENABLED = _MODE != "0"
    _VERIFY = _MODE == "verify"
    return _MODE


reload_mode()


def step_prologue_trim_enabled() -> bool:
    """Default on; ``TRTLLM_STEP_PROLOGUE_TRIM=0`` restores the original path."""
    return _ENABLED


def step_prologue_trim_verify() -> bool:
    """``TRTLLM_STEP_PROLOGUE_TRIM=verify``: trim on, plus a self-check at
    every elided write. Synchronizes the device at each check: a correctness
    mode, not a performance mode."""
    return _VERIFY


class StepPrologueTrimVerifyError(RuntimeError):
    pass


def verify_device_equals(dev: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """Verify mode: raise unless ``dev`` equals ``expected`` bit for bit."""
    torch.cuda.synchronize()
    if not torch.equal(dev.detach().cpu(), expected.detach().cpu()):
        raise StepPrologueTrimVerifyError(
            f"step-prologue trim: eliding the write of {what} would leave stale device data"
        )
    logger.info_once(
        f"{_ENV}=verify: elided writes of {what} are being verified",
        key=f"step-prologue-trim-verify-{what}",
    )


def _storage_key(dst: torch.Tensor) -> int:
    """Base address of ``dst``'s storage, without materializing a storage object."""
    return dst.data_ptr() - dst.storage_offset() * dst.element_size()


def _region_key(dst: torch.Tensor) -> Tuple[int, Tuple[int, ...], Tuple[int, ...], torch.dtype]:
    return (dst.data_ptr(), tuple(dst.shape), tuple(dst.stride()), dst.dtype)


class UnchangedCopyFilter:
    """Remembers, per device *storage*, the region and the host payload of the
    last copy issued through it.

    ``unchanged(dst, payload)`` is True only when the most recent write this
    filter saw on ``dst``'s storage was to exactly the same region (pointer,
    shape, strides, dtype) with a bit-identical payload. The caller then skips
    the copy: the device already holds those values. Keyed by the storage base
    pointer, not the tensor object, because attention-metadata buffers are
    shared across metadata instances (``Buffers.get_buffer`` by cache name);
    any write to the storage through the filter replaces the record, so a
    write of a different region (another metadata instance, another batch
    size) forces the next copy. The record holds a reference to the storage so
    its address cannot be recycled by the allocator while recorded.

    Correctness contract: every writer of a filtered device buffer either goes
    through ``record()`` or calls ``invalidate()``. The users in this repo are
    the only writers of their buffers (see the call sites).

    Verify mode: when ``unchanged()`` would return True it returns False
    instead (the caller copies) after cloning the device region; the
    following ``record()`` checks the copy left the region bit-identical.

    With the trim off ("0") nothing is ever recorded, and every method returns
    before touching torch, so that path is the original one.
    """

    def __init__(self) -> None:
        # storage base -> (storage, region, payload snapshot, extra)
        self._records: Dict[int, Tuple[Any, Hashable, Optional[torch.Tensor], Hashable]] = {}
        self._verify_pending: Dict[int, Tuple[Hashable, torch.Tensor]] = {}

    def fits(self, nbytes: int) -> bool:
        """Whether a payload of ``nbytes`` is within the compare cap."""
        return nbytes <= MAX_COMPARE_BYTES

    def unchanged(
        self, dst: torch.Tensor, payload: Optional[torch.Tensor] = None, extra: Hashable = None
    ) -> bool:
        # Cheapest checks first: a bare flag, then the size cap, then the
        # record lookup; the capture check runs only on an actual elision.
        if not _ENABLED or not self._records:
            return False
        if payload is not None and not self.fits(payload.numel() * payload.element_size()):
            return False
        key = _storage_key(dst)
        rec = self._records.get(key)
        if rec is None:
            return False
        _, region, snap, rec_extra = rec
        if region != _region_key(dst) or rec_extra != extra:
            return False
        if payload is None:
            same = snap is None
        else:
            same = snap is not None and snap.dtype == payload.dtype and torch.equal(snap, payload)
        if not same:
            return False
        # A copy issued under stream capture becomes a graph node that runs
        # on every replay; never elide it.
        if torch.cuda.is_current_stream_capturing():
            return False
        if _VERIFY:
            torch.cuda.synchronize()
            self._verify_pending[key] = (region, dst.detach().clone())
            return False
        return True

    def record(
        self, dst: torch.Tensor, payload: Optional[torch.Tensor] = None, extra: Hashable = None
    ) -> None:
        if not _ENABLED and not self._records and not self._verify_pending:
            return
        key = _storage_key(dst)
        pending = self._verify_pending.pop(key, None)
        if pending is not None and pending[0] == _region_key(dst):
            verify_device_equals(dst, pending[1], "a filtered table copy")
        if not _ENABLED or torch.cuda.is_current_stream_capturing():
            # A captured copy has not written anything yet; an untracked one
            # leaves the region unknown.
            self._records.pop(key, None)
            return
        if payload is not None and not self.fits(payload.numel() * payload.element_size()):
            self._records.pop(key, None)
            return
        snap = None
        if payload is not None:
            # Private (pageable) copy: the caller's staging buffer is
            # rewritten by the next prepare.
            snap = payload.detach().to("cpu", copy=True)
        # The record keeps the storage alive: while it exists the allocator
        # cannot hand this address to another buffer, so the key stays valid.
        self._records[key] = (dst.untyped_storage(), _region_key(dst), snap, extra)

    def invalidate(self, dst: torch.Tensor) -> None:
        if not self._records and not self._verify_pending:
            return
        key = _storage_key(dst)
        self._records.pop(key, None)
        self._verify_pending.pop(key, None)


# Process-wide filter for buffers whose writers live in different modules
# (KV-cache manager block offsets, DSA metadata tables).
GLOBAL_COPY_FILTER = UnchangedCopyFilter()
