# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Asynchronous host-to-device metadata uploads.

Bounded pinned staging, and skipping an upload whose payload the destination
already holds.
"""

import torch
from torch.utils.weak import WeakIdKeyDictionary

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned

_STAGING_RING = 2

# The last copy copy_if_changed() made into each storage, keyed by the
# storage's base tensor: views of one buffer share an entry, and the entry
# goes away with the buffer.
_last_copies = WeakIdKeyDictionary()


def _base(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor._base is None else tensor._base


def _same_payload(recorded, current) -> bool:
    if isinstance(recorded, torch.Tensor) or isinstance(current, torch.Tensor):
        return (
            isinstance(recorded, torch.Tensor)
            and isinstance(current, torch.Tensor)
            and recorded.dtype == current.dtype
            and recorded.shape == current.shape
            and torch.equal(recorded, current)
        )
    return recorded == current


def copy_if_changed(destination: torch.Tensor, values, *, token=None) -> bool:
    """Copy ``values`` into ``destination`` unless it already holds them.

    ``values`` is a host tensor or a sequence of integers. ``destination``
    already holds them when the last copy made here into its storage wrote
    the same elements with an equal payload. ``token``, when given, is
    compared in place of the payload; it must change whenever the payload
    does. A destination filtered here must not be written any other way
    without a ``forget_copy`` call. Returns whether a copy was issued.
    """
    if destination.numel() == 0:
        return False
    payload = values if isinstance(values, torch.Tensor) else tuple(values)
    key = payload if token is None else token
    region = (destination.data_ptr(), destination.dtype, destination.shape, destination.stride())
    base = _base(destination)
    last = _last_copies.get(base)
    if last is not None and last[0] == region and _same_payload(last[1], key):
        return False
    if not isinstance(payload, torch.Tensor):
        payload = torch.tensor(
            payload, dtype=destination.dtype, pin_memory=destination.is_cuda and prefer_pinned()
        )
    destination.copy_(payload, non_blocking=destination.is_cuda)
    _last_copies[base] = (region, key.clone() if isinstance(key, torch.Tensor) else key)
    return True


def forget_copy(destination: torch.Tensor) -> None:
    """Drop what ``copy_if_changed`` recorded for ``destination``'s storage.

    Call it after writing that storage any other way.
    """
    _last_copies.pop(_base(destination), None)


def copy_host_to_device(store: dict, key, destination: torch.Tensor, value: torch.Tensor) -> None:
    """Asynchronously copy a host tensor into ``destination`` through pinned staging.

    Pageable or strided sources make ``copy_(non_blocking=True)`` a synchronous
    staged copy inside the driver (a strided source is first materialized as a
    pageable temporary), so the value is first written into a contiguous pinned
    buffer. Each key owns a small ring of buffers with a CUDA event recorded
    after every copy: a buffer is only rewritten once the copy that last read
    it has completed, so the host may run ahead of the device (overlap
    scheduling, chunked prefill) without corrupting a copy still in flight.
    """
    if value.numel() == 0:
        return
    if (
        not destination.is_cuda
        or not prefer_pinned()
        or (value.is_pinned() and value.is_contiguous())
    ):
        # Host destinations (tests) and already pinned sources need no staging.
        destination.copy_(value, non_blocking=destination.is_cuda)
        return
    slot = (key, value.dtype)
    ring = store.get(slot)
    if ring is None:
        ring = store[slot] = {
            "buffers": [None] * _STAGING_RING,
            "events": [None] * _STAGING_RING,
            "next": 0,
        }
    index = ring["next"]
    ring["next"] = (index + 1) % _STAGING_RING
    event = ring["events"][index]
    if event is not None:
        event.synchronize()
    buffer = ring["buffers"][index]
    if buffer is None or buffer.numel() < value.numel():
        buffer = maybe_pin_memory(
            torch.empty(
                max(value.numel(), 2 * buffer.numel() if buffer is not None else 0),
                dtype=value.dtype,
                device="cpu",
            )
        )
        ring["buffers"][index] = buffer
    view = buffer[: value.numel()].view(value.shape)
    view.copy_(value)
    destination.copy_(view, non_blocking=True)
    if event is None:
        event = ring["events"][index] = torch.cuda.Event()
    event.record(torch.cuda.current_stream(destination.device))
