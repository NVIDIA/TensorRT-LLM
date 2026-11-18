# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-position KDA checkpoints: byte layout, MR8 flat views, seed / commit and slot indices.

This module depends on PyTorch only. :mod:`.cache` binds it to the KV cache manager's pools.

One KDA layer keeps, per recurrent-state slot, one buffer of role ``kda_ckpt``::

    [256 B header][num_checkpoints x state [H, V, K]][conv page [conv_dim, conv_width]][pad to 256 B]

* Checkpoint ``t`` is the recurrent state after verify position ``t`` (``num_checkpoints`` =
  ``1 + num_spec``); its dtype is the manager's SSM state dtype, FP32 or BF16.
* The conv page holds raw BF16 convolution inputs, dimension-first; ``conv_width`` =
  ``conv_window + num_spec``. Columns ``[a, a + conv_window)`` are the committed window after
  ``a`` accepted drafts.

The MR8 KDA kernels index one state tensor and one conv tensor with int32
``slots[b, t]`` (the conv page at ``slots[b, 0]``) and treat ``slot <= 0`` as padding. Both flat
views here (:func:`kda_checkpoint_views`) have a row stride of ``U`` bytes, the largest power of
two from 16 to 256 that divides the slot stride ``P``, the buffer address and the state size, so

    ``slots[b, t] = 256 / U + s * P / U + t * state_bytes / U``

addresses checkpoint ``t`` and the conv page of slot ``s``. The header keeps every real index
positive without a view that starts before the buffer. With the 256-byte header and padding,
``U`` reaches 256 where the slot's other buffers allow it; a pool of ``slots * P`` bytes then
needs ``slots * P / U`` rows, which int32 covers up to 512 GiB at ``U`` = 256 (32 GiB at 16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

KDA_CHECKPOINT_ROLE = "kda_ckpt"
# Bytes of the per-slot accepted-draft count buffer. It is as large as the checkpoint
# granule, so a slot that coalesces it with the checkpoint payloads keeps a stride that
# is a multiple of the largest row stride.
KDA_CHECKPOINT_COUNT_BYTES = 256
# Smallest row stride of the flat kernel views (the kernels' 16-byte state alignment).
_INDEX_BYTES = 16
# Leading unused bytes of a slot's payload, its size granule and the largest row stride.
_HEADER_BYTES = 256
_INT32_MAX = 2**31 - 1

KDA_CHECKPOINT_STATE_DTYPES = (torch.float32, torch.bfloat16)
KDA_CHECKPOINT_CONV_DTYPES = (torch.bfloat16,)


@dataclass(frozen=True)
class KDACheckpointLayout:
    """One KDA layer's ``kda_ckpt`` payload of a state slot (see the module docstring)."""

    num_checkpoints: int
    state_shape: tuple[int, int, int]
    state_dtype: torch.dtype
    conv_dim: int
    conv_width: int
    conv_dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.state_dtype not in KDA_CHECKPOINT_STATE_DTYPES:
            raise TypeError(
                f"KDA checkpoints keep FP32 or BF16 recurrent state, not {self.state_dtype}"
            )
        if self.conv_dtype not in KDA_CHECKPOINT_CONV_DTYPES:
            raise TypeError(f"KDA checkpoints keep BF16 raw conv inputs, not {self.conv_dtype}")
        if self.num_checkpoints <= 0 or self.conv_dim <= 0 or self.conv_width <= 0:
            raise ValueError("KDA checkpoint dimensions must be positive")
        if len(self.state_shape) != 3 or any(dim <= 0 for dim in self.state_shape):
            raise ValueError("KDA recurrent state needs positive [heads, V, K] dimensions")

    @property
    def state_bytes(self) -> int:
        return self.state_dtype.itemsize * math.prod(self.state_shape)

    @property
    def conv_offset(self) -> int:
        return _HEADER_BYTES + self.num_checkpoints * self.state_bytes

    @property
    def conv_bytes(self) -> int:
        return self.conv_dtype.itemsize * self.conv_dim * self.conv_width

    @property
    def nbytes(self) -> int:
        end = self.conv_offset + self.conv_bytes
        return -(-end // _HEADER_BYTES) * _HEADER_BYTES


def kda_checkpoint_layout(
    num_spec: int,
    num_heads: int,
    head_dim: int,
    conv_window: int,
    state_dtype: torch.dtype,
    conv_dtype: torch.dtype,
) -> KDACheckpointLayout:
    """The payload of one layer: ``1 + num_spec`` states ``[num_heads, head_dim, head_dim]`` and
    one conv page ``[3 * num_heads * head_dim, conv_window + num_spec]`` (q | k | v)."""
    return KDACheckpointLayout(
        num_checkpoints=1 + num_spec,
        state_shape=(num_heads, head_dim, head_dim),
        state_dtype=state_dtype,
        conv_dim=3 * num_heads * head_dim,
        conv_width=conv_window + num_spec,
        conv_dtype=conv_dtype,
    )


def estimate_checkpoint_bytes_per_slot(layout: KDACheckpointLayout, num_local_layers: int) -> int:
    """Bytes one state slot holds for ``num_local_layers`` layers' payloads and the count."""
    if num_local_layers <= 0:
        return 0
    return num_local_layers * layout.nbytes + KDA_CHECKPOINT_COUNT_BYTES


@dataclass(frozen=True)
class KDACheckpointViews:
    """Views of one layer's ``kda_ckpt`` buffer.

    ``ckpt`` [slots, num_checkpoints, H, V, K] and ``vconv`` [slots, conv_dim, conv_width] are the
    typed per-slot views; ``state_flat`` [rows, H, V, K] and ``conv_flat`` [conv_rows, conv_dim,
    conv_width] are the MR8 kernel views with a row stride of ``unit_bytes``. All four share the
    buffer's storage.
    """

    ckpt: torch.Tensor
    vconv: torch.Tensor
    state_flat: torch.Tensor
    conv_flat: torch.Tensor
    unit_bytes: int
    slot_units: int
    state_units: int
    header_units: int


def checkpoint_unit(slot_bytes: int, address: int, state_bytes: int) -> int:
    """The flat views' row stride: the largest power of two from 16 to 256 dividing the slot
    stride, the buffer address and the state size (16 when none of them is larger)."""
    unit = _HEADER_BYTES
    while unit > _INDEX_BYTES and (slot_bytes % unit or address % unit or state_bytes % unit):
        unit //= 2
    return unit


def kda_checkpoint_views(raw: torch.Tensor, layout: KDACheckpointLayout) -> KDACheckpointViews:
    """The typed per-slot views and the MR8 flat views of a ``kda_ckpt`` buffer.

    ``raw`` is the pool's ``[slots, payload_bytes]`` uint8 view; its slot stride may exceed the
    payload (other buffers of the slot). Raises ValueError when the buffer is too small or
    misaligned, or when the flat rows do not fit int32 indices.
    """
    if raw.dtype != torch.uint8 or raw.dim() != 2 or raw.stride(1) != 1:
        raise ValueError("kda_ckpt buffer must be a [slots, bytes] uint8 view")
    slots, slot_bytes = raw.shape[0], raw.stride(0)
    if slots <= 0:
        raise ValueError("kda_ckpt buffer has no slots")
    unit = checkpoint_unit(slot_bytes, raw.data_ptr(), layout.state_bytes)
    if raw.shape[1] < layout.nbytes or slot_bytes % unit or raw.data_ptr() % unit:
        raise ValueError(
            f"kda_ckpt buffer [{raw.shape[0]}, {raw.shape[1]}] (slot stride {slot_bytes} B) "
            f"cannot hold {layout.nbytes} B per slot at a {unit}-byte alignment"
        )
    if layout.state_bytes % unit:
        raise ValueError(f"KDA checkpoint states must be a multiple of {unit} bytes")
    n, (h, v, k) = layout.num_checkpoints, layout.state_shape
    state_isz, conv_isz = layout.state_dtype.itemsize, layout.conv_dtype.itemsize
    if (layout.conv_offset - _HEADER_BYTES) % conv_isz:
        raise ValueError("KDA checkpoint conv page is not aligned to its dtype")
    slot_units, state_units = slot_bytes // unit, layout.state_bytes // unit
    header_units = _HEADER_BYTES // unit
    rows = (slots - 1) * slot_units + header_units + (n - 1) * state_units + 1
    if rows > _INT32_MAX:
        raise ValueError(
            f"{slots} KDA checkpoint slots of {slot_bytes} B at a {unit}-byte row stride exceed "
            "the kernels' int32 slot indices"
        )
    states = raw[:, _HEADER_BYTES : layout.conv_offset].view(layout.state_dtype)
    ckpt = states.unflatten(-1, (n, h, v, k))
    page = raw[:, layout.conv_offset : layout.conv_offset + layout.conv_bytes]
    vconv = page.view(layout.conv_dtype).unflatten(-1, (layout.conv_dim, layout.conv_width))
    # Row r of a flat view starts at byte r * unit of the buffer: slot s's checkpoint t at row
    # header + s * slot_units + t * state_units, its conv page at row header + s * slot_units.
    typed = raw.view(layout.state_dtype)
    state_flat = typed.as_strided(
        (rows, h, v, k), (unit // state_isz, v * k, k, 1), typed.storage_offset()
    )
    # The kernels read a conv page only at a request's first index (the slot's row), so the conv
    # view ends at the last slot's row; one more row would overrun the buffer.
    conv_rows = (slots - 1) * slot_units + header_units + 1
    typed = raw.view(layout.conv_dtype)
    conv_flat = typed.as_strided(
        (conv_rows, layout.conv_dim, layout.conv_width),
        (unit // conv_isz, layout.conv_width, 1),
        typed.storage_offset() + (layout.conv_offset - _HEADER_BYTES) // conv_isz,
    )
    return KDACheckpointViews(
        ckpt=ckpt,
        vconv=vconv,
        state_flat=state_flat,
        conv_flat=conv_flat,
        unit_bytes=unit,
        slot_units=slot_units,
        state_units=state_units,
        header_units=header_units,
    )


def checkpoint_position_rows(views: KDACheckpointViews, device: torch.device) -> torch.Tensor:
    """int64 ``[num_checkpoints]``: the flat row of each checkpoint of slot 0."""
    positions = torch.arange(views.ckpt.shape[1], dtype=torch.int64, device=device)
    return views.header_units + positions * views.state_units


def seed_kda_checkpoints(
    ckpt: torch.Tensor,
    vconv: torch.Tensor,
    ssm_pool: torch.Tensor,
    conv_pool: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """Checkpoint 0 and the committed conv window (columns ``[0, W)``) of ``slots`` (int64)
    from the live pools."""
    if slots.numel() == 0:
        return
    window = conv_pool.shape[-1]
    ckpt[slots, 0] = ssm_pool.index_select(0, slots).to(ckpt.dtype)
    vconv[slots, :, :window] = conv_pool.index_select(0, slots).to(vconv.dtype)


def stack_layer_views(views: list[torch.Tensor]) -> torch.Tensor | None:
    """One ``[num_layers, *shape]`` view of per-layer views of one storage.

    The views must agree in dtype, device, shape and strides and start a constant positive
    number of elements apart; otherwise (or for an empty list) the result is None. The
    runtime's ``_stack_state_views`` stacks pool views that are separate tensors over one CUDA
    allocation; this one covers views that share a storage object (CPU pools).
    """
    if not views:
        return None
    ref = views[0]
    storage = ref.untyped_storage().data_ptr()
    for view in views[1:]:
        if (
            view.dtype != ref.dtype
            or view.device != ref.device
            or view.shape != ref.shape
            or view.stride() != ref.stride()
            or view.untyped_storage().data_ptr() != storage
        ):
            return None
    step = views[1].storage_offset() - ref.storage_offset() if len(views) > 1 else 0
    if len(views) > 1 and step <= 0:
        return None
    if any(
        view.storage_offset() != ref.storage_offset() + layer * step
        for layer, view in enumerate(views)
    ):
        return None
    return ref.as_strided((len(views), *ref.shape), (step, *ref.stride()), ref.storage_offset())


def seed_kda_checkpoints_stacked(
    ckpt0: torch.Tensor,
    vconv_window: torch.Tensor,
    ssm_pools: torch.Tensor,
    conv_pools: torch.Tensor,
    slots: torch.Tensor,
) -> None:
    """:func:`seed_kda_checkpoints` of every layer at once, one gather and one scatter per
    buffer kind.

    ``ckpt0`` [layers, slots, H, V, K] is checkpoint 0 of every slot, ``vconv_window`` [layers,
    slots, conv_dim, W] the conv columns ``[0, W)``, ``ssm_pools`` / ``conv_pools`` the live
    pools, all stacked over the layers (:func:`stack_layer_views`). ``slots`` is int64 and
    holds no duplicates.
    """
    if slots.numel() == 0:
        return
    ckpt0.index_copy_(1, slots, ssm_pools.index_select(1, slots).to(ckpt0.dtype))
    vconv_window.index_copy_(1, slots, conv_pools.index_select(1, slots).to(vconv_window.dtype))


def commit_kda_checkpoints(
    ckpt: torch.Tensor,
    vconv: torch.Tensor,
    ssm_pool: torch.Tensor,
    conv_pool: torch.Tensor,
    slots: torch.Tensor,
    accepted_drafts: torch.Tensor,
) -> None:
    """The live pools of ``slots`` (int64) from their current checkpoints: checkpoint
    ``a = accepted_drafts[slot]`` and conv columns ``[a, a + W)``."""
    if slots.numel() == 0:
        return
    window = conv_pool.shape[-1]
    position = accepted_drafts.index_select(0, slots).long()
    ssm_pool.index_copy_(0, slots, ckpt[slots, position].to(ssm_pool.dtype))
    columns = position[:, None, None] + torch.arange(window, device=slots.device)
    pages = vconv.index_select(0, slots)
    columns = columns.expand(-1, pages.shape[1], -1)
    conv_pool.index_copy_(0, slots, pages.gather(2, columns).to(conv_pool.dtype))


def checkpoint_indices(
    state_indices: torch.Tensor,
    accepted_drafts: torch.Tensor,
    position_rows: torch.Tensor,
    slot_units: int,
    out_slots: torch.Tensor | None = None,
    out_accepted: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The MR8 ``(spec_state_indices [n, num_checkpoints], num_accepted_tokens [n])`` int32 of
    the requests whose manager slots are ``state_indices`` [n].

    ``accepted_drafts`` is the per-slot count of drafts accepted by the last verify (it excludes
    the target token; the kernels' count includes it). ``position_rows`` is
    :func:`checkpoint_position_rows`. Device operations only, so the call can be captured in a
    CUDA graph; ``out_slots`` / ``out_accepted`` are written in place when given. Every entry of
    ``state_indices`` must be a live slot: the rows are always positive, so no entry is treated
    as padding.
    """
    base = state_indices.long()
    rows = base[:, None] * slot_units + position_rows
    counts = accepted_drafts.index_select(0, base) + 1
    if out_slots is None:
        out_slots = rows.to(torch.int32)
    else:
        if out_slots.dtype != torch.int32 or out_slots.shape != rows.shape:
            raise ValueError(
                f"out_slots must be int32 {tuple(rows.shape)}, got "
                f"{out_slots.dtype} {tuple(out_slots.shape)}"
            )
        out_slots.copy_(rows)
    if out_accepted is None:
        out_accepted = counts.to(torch.int32)
    else:
        if out_accepted.dtype != torch.int32 or out_accepted.shape != counts.shape:
            raise ValueError(
                f"out_accepted must be int32 {tuple(counts.shape)}, got "
                f"{out_accepted.dtype} {tuple(out_accepted.shape)}"
            )
        out_accepted.copy_(counts)
    return out_slots, out_accepted


__all__ = [
    "KDA_CHECKPOINT_COUNT_BYTES",
    "KDA_CHECKPOINT_ROLE",
    "KDA_CHECKPOINT_STATE_DTYPES",
    "KDACheckpointLayout",
    "KDACheckpointViews",
    "checkpoint_indices",
    "checkpoint_position_rows",
    "checkpoint_unit",
    "commit_kda_checkpoints",
    "estimate_checkpoint_bytes_per_slot",
    "kda_checkpoint_layout",
    "kda_checkpoint_views",
    "seed_kda_checkpoints",
    "seed_kda_checkpoints_stacked",
    "stack_layer_views",
]
