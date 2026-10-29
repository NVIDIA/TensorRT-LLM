# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skip host-to-device metadata copies whose values the destination already holds.

Steady decode re-uploads many small per-step metadata tensors whose values do not change (slot indices, query
lengths, prompt lengths, page tables between page boundaries). Each upload is a separate copy on the compute stream,
serialized between the decode graphs. An ``H2DSkipCache`` keeps one host shadow per destination buffer (the last
values written through it, and the view they were written to) and skips the copy when the same view receives the same
values again.

Each engine that armed ``RuntimeHooks.skip_unchanged_h2d`` owns one cache and makes it the active one for its forward
(``active``); ``copy_h2d_if_changed`` acts on the calling context's active cache and, with none, is
``dst.copy_(src, non_blocking=True)``. The attention metadata that calls it has no engine to ask, which is why the
cache travels in the context rather than as an argument.

Route a copy through it only for a destination that no other code writes (no device kernel, no other copy path, no other
engine): the shadow describes the device contents only under that invariant. Pooled metadata buffers satisfy it because
an engine that arms the skip gives its CUDA graph metadata a buffer pool of its own
(``RuntimeHooks.skips_unchanged_metadata_uploads``); eager metadata allocates its own buffers. A write through a
different view of the same buffer replaces the shadow, so overlapping views never skip on stale state.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator, Optional

import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer

from . import step_inputs


class H2DSkipCache:
    """One engine's host shadows of its skip-routed destinations (see the module docstring)."""

    def __init__(self) -> None:
        # Device storage address -> (the storage, kept alive so the address is not reused; the view key; the host
        # values). Keyed by storage rather than by view base: tensors made under inference mode do not record their
        # base.
        self._shadow: dict[int, tuple[object, tuple, DeviceCopyBuffer]] = {}
        # Device storage address -> (the storage, kept alive as above; the view key; (first row, value) of the view's
        # last tail fill).
        self._tails: dict[int, tuple[object, tuple, tuple[int, int]]] = {}

    def clear(self) -> None:
        """Forget every shadow, so the next copy to each destination is issued (e.g. after a discarded step)."""
        self._shadow.clear()
        self._tails.clear()

    def copy(self, dst: torch.Tensor, src: torch.Tensor, pack: bool) -> bool:
        storage = dst.untyped_storage()
        addr = storage.data_ptr()
        if src.is_cuda or not dst.is_cuda or torch.cuda.is_current_stream_capturing():
            self._shadow.pop(addr, None)
            dst.copy_(src, non_blocking=True)
            return True
        key = (dst.storage_offset(), tuple(dst.shape), tuple(dst.stride()), dst.dtype)
        entry = self._shadow.get(addr)
        if entry is not None and entry[1] == key and entry[2].matches(src, exact=True):
            return False
        buffer = entry[2] if entry is not None and entry[1] == key else DeviceCopyBuffer(dst)
        buffer.invalidate()
        self._shadow.pop(addr, None)
        if pack:
            step_inputs.copy_(dst, src)
        else:
            dst.copy_(src, non_blocking=True)
        buffer.record(src)
        self._shadow[addr] = (storage, key, buffer)
        return True

    def fill_tail(self, buf: torch.Tensor, start: int, value: int) -> None:
        storage = buf.untyped_storage()
        addr = storage.data_ptr()
        if start >= buf.shape[0]:
            # The live rows cover the whole view: the previous tail fill is overwritten.
            self._tails.pop(addr, None)
            return
        if torch.cuda.is_current_stream_capturing():
            self._tails.pop(addr, None)
            buf[start:].fill_(value)
            return
        key = (buf.storage_offset(), tuple(buf.shape), tuple(buf.stride()), buf.dtype)
        entry = self._tails.get(addr)
        if entry is not None and entry[1] == key and entry[2] == (start, value):
            return
        buf[start:].fill_(value)
        self._tails[addr] = (storage, key, (start, value))


_active: ContextVar[Optional[H2DSkipCache]] = ContextVar("trtllm_h2d_skip_cache", default=None)


@contextlib.contextmanager
def active(cache: Optional[H2DSkipCache]) -> Iterator[None]:
    """Make ``cache`` the calling context's cache (None: no skipping) for the duration."""
    token = _active.set(cache)
    try:
        yield
    finally:
        _active.reset(token)


def enabled() -> bool:
    """Whether the calling context skips unchanged copies."""
    return _active.get() is not None


def copy_h2d_if_changed(dst: torch.Tensor, src: torch.Tensor, pack: bool = False) -> bool:
    """``dst.copy_(src, non_blocking=True)`` unless the active cache knows ``dst`` holds ``src``; True if copied.

    ``pack``: the copy may join an open ``step_inputs`` pack, i.e. land only at the pack's flush (the graph replay).
    Pass it only for destinations no device work reads before that replay.
    """
    cache = _active.get()
    if cache is None:
        dst.copy_(src, non_blocking=True)
        return True
    return cache.copy(dst, src, pack)


def fill_tail_if_changed(buf: torch.Tensor, start: int, value: int) -> None:
    """``buf[start:].fill_(value)`` unless the active cache's last tail fill of ``buf`` was the same.

    Call it after every write of the live rows ``buf[:start]``, including one that covers the whole of ``buf``
    (``start == len(buf)``, which fills nothing): a live-row write past the recorded start overwrites part of the
    tail. The skip is sound only while nothing else writes ``buf`` past ``start``.
    """
    cache = _active.get()
    if cache is None:
        if start < buf.shape[0]:
            buf[start:].fill_(value)
        return
    cache.fill_tail(buf, start, value)
