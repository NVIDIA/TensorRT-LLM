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
engine, no graph replay): the shadow describes the device contents only under that invariant. CUDA graph metadata
buffers do not satisfy it: every graph key's metadata takes them from one pool, which hands the same block to each key
that asks for the same name, and a captured graph's device writes into a block run again at each of its replays. Those
destinations use ``copy_h2d_pool_shared`` / ``fill_tail_pool_shared``, which never skip. A write through a different
view of the same buffer replaces the shadow, so overlapping views never skip on stale state.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator, Optional

import torch

from . import step_inputs


class H2DSkipCache:
    """One engine's host shadows of its skip-routed destinations (see the module docstring).

    ``changed_span``: a packed copy into a 2-D view with contiguous rows whose shadow covers the same rows (same
    first element and row stride) uploads, per row, only the span from its first to its last changed column, plus any
    columns the shadow does not cover. A page table that gained one page then stages a few bytes instead of every
    padded row, so the pack still fits the decode graph's prologue window.
    """

    # A changed-span upload is used only while it moves at most this fraction of the view's bytes.
    CHANGED_SPAN_MAX_FRACTION = 0.5

    def __init__(self, changed_span: bool = False) -> None:
        self.changed_span = changed_span
        # Device storage address -> (the storage, kept alive so the address is not reused; the view key; the host
        # values). Keyed by storage rather than by view base: tensors made under inference mode do not record their
        # base.
        self._shadow: dict[int, tuple[object, tuple, torch.Tensor]] = {}
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
        if (
            entry is not None
            and entry[1] == key
            and entry[2].dtype == src.dtype
            and entry[2].shape == src.shape
            and torch.equal(entry[2], src)
        ):
            return False
        if not (
            pack
            and self.changed_span
            and entry is not None
            and self._copy_changed_span(dst, src, entry[1:])
        ):
            if pack:
                step_inputs.copy_(dst, src)
            else:
                dst.copy_(src, non_blocking=True)
        self._shadow[addr] = (storage, key, src.clone())
        return True

    def _copy_changed_span(self, dst: torch.Tensor, src: torch.Tensor, shadow: tuple) -> bool:
        """Stage only the changed span of each row of ``dst`` (see the class docstring); False if not applicable."""
        (offset, shape, stride, dtype), old = shadow
        if (
            dst.dim() != 2
            or dst.stride(1) != 1
            or offset != dst.storage_offset()
            or len(shape) != 2
            or shape[0] != dst.shape[0]
            or stride != tuple(dst.stride())
            or dtype != dst.dtype
            or old.dtype != src.dtype
            or tuple(src.shape) != tuple(dst.shape)
            or not step_inputs.is_open()
        ):
            return False
        cols = dst.shape[1]
        known = min(shape[1], cols)
        changed = src[:, :known] != old[:, :known]
        if cols > known:
            # Columns the shadow does not cover hold unknown device values.
            changed = torch.cat([changed, changed.new_ones(dst.shape[0], cols - known)], dim=1)
        rows = changed.any(dim=1).nonzero().flatten().tolist()
        spans = []
        for row in rows:
            columns = changed[row].nonzero().flatten()
            spans.append((row, int(columns[0]), int(columns[-1]) + 1))
        if sum(hi - lo for _, lo, hi in spans) > self.CHANGED_SPAN_MAX_FRACTION * dst.numel():
            return False
        for row, lo, hi in spans:
            step_inputs.copy_(dst[row, lo:hi], src[row, lo:hi])
        return True

    def forget(self, dst: torch.Tensor) -> None:
        """Drop the shadow and tail record of ``dst``'s storage (it is about to be written outside the cache)."""
        addr = dst.untyped_storage().data_ptr()
        self._shadow.pop(addr, None)
        self._tails.pop(addr, None)

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

# Kill switch for A/B only: True routes the pool-shared metadata destinations (``copy_h2d_pool_shared``,
# ``fill_tail_pool_shared``) through the skip cache again, as before. False keeps them always written.
SKIP_POOL_SHARED_UPLOADS = False


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


def copy_h2d_pool_shared(dst: torch.Tensor, src: torch.Tensor, pack: bool = False) -> bool:
    """``copy_h2d_if_changed`` for a destination in a CUDA graph metadata buffer pool: always issued.

    Every graph key's metadata takes its buffers from one pool, which hands the same block to each key that asks
    for the same buffer name, and the captured graphs write some of those blocks on the device at every replay. The
    host shadow therefore does not describe the device contents, so such a copy is never skipped. ``pack``: as for
    ``copy_h2d_if_changed``, the copy may join the open ``step_inputs`` pack when a cache is active.
    """
    cache = _active.get()
    if cache is None:
        dst.copy_(src, non_blocking=True)
        return True
    if SKIP_POOL_SHARED_UPLOADS:
        return cache.copy(dst, src, pack)
    cache.forget(dst)
    if pack:
        step_inputs.copy_(dst, src)
    else:
        dst.copy_(src, non_blocking=True)
    return True


def fill_tail_pool_shared(buf: torch.Tensor, start: int, value: int) -> None:
    """``buf[start:].fill_(value)`` for a buffer in a CUDA graph metadata buffer pool: always issued.

    See ``copy_h2d_pool_shared``: another graph key's replay can write ``buf`` past ``start``.
    """
    cache = _active.get()
    if cache is not None and SKIP_POOL_SHARED_UPLOADS:
        cache.fill_tail(buf, start, value)
        return
    if cache is not None:
        cache.forget(buf)
    if start < buf.shape[0]:
        buf[start:].fill_(value)
