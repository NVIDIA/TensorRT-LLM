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
"""Batched seeding of the KDA replay caches of newly transferred requests.

A disaggregated generation worker receives the committed ``W - 1`` convolution
window of every KDA layer from the context worker. Before the request's first
verify step, its replay caches (the extended convolution windows and the
``k``/``g``/``v``/``beta`` histories) are cleared and the committed window is
copied in. Done one layer at a time that is ``11 * num_layers`` tiny PyTorch ops
per admission (seven clears, one gather and three scatters per layer), queued on
the stream in front of the next decode step.

:func:`seed_replay_caches` does the same with one Triton launch per replay
buffer, covering all layers and slots. The replay buffers and conv states of a
layer are views into the cache manager's pools; a :class:`ReplaySeedPlan`
records their base addresses in device tables once and stays valid for as long
as those views do.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

_TORCH_TO_TL_DTYPE = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}

# Elements of the unit-stride dimension handled by one program.
_BLOCK_FAST = 512

CONV_BUFFER_NAMES = ("kda_conv_q", "kda_conv_k", "kda_conv_v")

# Admissions seeded by a plan so far, logged after 1, 10, 100, ... of them: building a plan at
# pool allocation only says the batched path is available, these lines say it ran on live admissions.
_seeded_admissions = 0
_seeded_slots = 0
_next_log_admissions = 1


BATCHED_REPLAY_SEED_ENV = "TRTLLM_KDA_BATCHED_REPLAY_SEED"


def batched_replay_seed_enabled() -> bool:
    """TRTLLM_KDA_BATCHED_REPLAY_SEED=1 selects the batched seed; unset or 0 keeps the per-layer loop."""
    value = os.environ.get(BATCHED_REPLAY_SEED_ENV, "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{BATCHED_REPLAY_SEED_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


@triton.jit
def _seed_replay_buffer_kernel(
    dst_ptrs,
    src_ptrs,
    slot_idx_ptr,
    dst_slot_stride,
    dst_slow_stride,
    src_slot_stride,
    src_fast_stride,
    src_slow_stride,
    src_fast_offset,
    num_fast,
    num_slow,
    DST_DTYPE: tl.constexpr,
    SRC_DTYPE: tl.constexpr,
    COPY: tl.constexpr,
    COMMITTED: tl.constexpr,
    BLOCK_FAST: tl.constexpr,
    BLOCK_SLOW: tl.constexpr,
):
    """Clear one replay buffer of one layer and slot; with COPY also fill its window.

    Every destination view is ``[slots, slow, fast]`` with a unit stride along
    ``fast``. With ``COPY`` the first ``COMMITTED`` slow positions are read from the
    layer's conv state (channel ``src_fast_offset + fast``) and the rest is zeroed.
    Programs are laid out as ``(layer, slot, fast tile)``.
    """
    layer = tl.program_id(0)
    slot = tl.load(slot_idx_ptr + tl.program_id(1)).to(tl.int64)
    fast = tl.program_id(2) * BLOCK_FAST + tl.arange(0, BLOCK_FAST)
    slow = tl.arange(0, BLOCK_SLOW)
    mask = (fast[None, :] < num_fast) & (slow[:, None] < num_slow)
    dst_base = tl.load(dst_ptrs + layer).to(tl.pointer_type(DST_DTYPE))
    dst = dst_base + slot * dst_slot_stride + slow[:, None] * dst_slow_stride + fast[None, :]
    if COPY:
        src_base = tl.load(src_ptrs + layer).to(tl.pointer_type(SRC_DTYPE))
        src = (
            src_base
            + slot * src_slot_stride
            + (src_fast_offset + fast[None, :]) * src_fast_stride
            + slow[:, None] * src_slow_stride
        )
        values = tl.load(src, mask=mask & (slow[:, None] < COMMITTED), other=0.0)
        tl.store(dst, values.to(DST_DTYPE), mask=mask)
    else:
        tl.store(dst, tl.zeros([BLOCK_SLOW, BLOCK_FAST], dtype=DST_DTYPE), mask=mask)


@dataclass(frozen=True)
class _BufferPlan:
    """One replay buffer across all layers, as ``[slots, slow, fast]`` views."""

    ptrs: torch.Tensor  # [num_layers] int64 base addresses
    dtype: torch.dtype
    slot_stride: int
    slow_stride: int
    num_fast: int
    num_slow: int
    conv_channel_offset: Optional[int]  # first conv-state channel (conv windows), else None


@dataclass(frozen=True)
class ReplaySeedPlan:
    """Device address tables of the replay buffers and conv states of all layers."""

    num_layers: int
    conv_ptrs: torch.Tensor  # [num_layers] int64 base addresses of the conv states
    conv_dtype: torch.dtype
    conv_slot_stride: int
    conv_channel_stride: int
    conv_window_stride: int
    committed_window: int
    buffers: Tuple[_BufferPlan, ...]


def _uniform_layout(tensors: Sequence[torch.Tensor]) -> bool:
    first = tensors[0]
    return all(
        t.dtype == first.dtype
        and t.shape == first.shape
        and t.stride() == first.stride()
        and t.device == first.device
        for t in tensors
    )


def _conv_window_plan(
    views: Sequence[torch.Tensor], channel_offset: int, section_dim: int, committed_window: int
) -> Optional[_BufferPlan]:
    # Replay conv windows are [slots, channels, committed + draft] views of storage
    # laid out [slots, window, channels]: the channel dimension has the unit stride.
    first = views[0]
    if first.stride(1) != 1 or first.shape[1] != section_dim or first.shape[2] < committed_window:
        return None
    return _BufferPlan(
        ptrs=_pointer_table(views),
        dtype=first.dtype,
        slot_stride=first.stride(0),
        slow_stride=first.stride(2),
        num_fast=first.shape[1],
        num_slow=first.shape[2],
        conv_channel_offset=channel_offset,
    )


def _history_plan(views: Sequence[torch.Tensor]) -> Optional[_BufferPlan]:
    # History caches are [slots, draft, features] views with unit-stride features.
    first = views[0]
    if first.stride(2) != 1:
        return None
    return _BufferPlan(
        ptrs=_pointer_table(views),
        dtype=first.dtype,
        slot_stride=first.stride(0),
        slow_stride=first.stride(1),
        num_fast=first.shape[2],
        num_slow=first.shape[1],
        conv_channel_offset=None,
    )


def _pointer_table(views: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.tensor([v.data_ptr() for v in views], dtype=torch.int64, device=views[0].device)


def build_replay_seed_plan(
    replay_layers: Sequence[Dict[str, torch.Tensor]],
    conv_states: Sequence[torch.Tensor],
    section_offsets: Sequence[int],
    committed_window: int,
) -> Optional[ReplaySeedPlan]:
    """Record the layers' buffer addresses, or return None when they cannot be batched.

    ``replay_layers[i]`` maps a replay buffer name to the layer's view: ``[slots,
    channels, window]`` for the Q/K/V conv windows and ``[slots, draft, features]``
    for the history caches. ``conv_states[i]`` is the layer's ``[slots, channels,
    committed_window]`` transferred state with the Q, K and V sections between
    ``section_offsets``. The plan needs every layer to share one layout and dtype,
    and every buffer to be device-resident with a unit-stride feature dimension.
    """
    num_layers = len(replay_layers)
    if num_layers == 0 or len(conv_states) != num_layers or len(section_offsets) != 4:
        return None
    names = list(replay_layers[0].keys())
    if any(list(layer.keys()) != names for layer in replay_layers):
        return None
    if any(name not in names for name in CONV_BUFFER_NAMES):
        return None
    conv0 = conv_states[0]
    if (
        conv0.dim() != 3
        or conv0.dtype not in _TORCH_TO_TL_DTYPE
        or not _uniform_layout(conv_states)
    ):
        return None
    section_dims = {section_offsets[i + 1] - section_offsets[i] for i in range(3)}
    if len(section_dims) != 1:
        return None
    (section_dim,) = section_dims

    buffers: List[_BufferPlan] = []
    for name in names:
        views = [layer[name] for layer in replay_layers]
        first = views[0]
        if (
            first.dim() != 3
            or first.device != conv0.device
            or first.dtype not in _TORCH_TO_TL_DTYPE
            or not _uniform_layout(views)
        ):
            return None
        if name in CONV_BUFFER_NAMES:
            plan = _conv_window_plan(
                views,
                section_offsets[CONV_BUFFER_NAMES.index(name)],
                section_dim,
                committed_window,
            )
        else:
            plan = _history_plan(views)
        if plan is None:
            return None
        buffers.append(plan)
    return ReplaySeedPlan(
        num_layers=num_layers,
        conv_ptrs=_pointer_table(conv_states),
        conv_dtype=conv0.dtype,
        conv_slot_stride=conv0.stride(0),
        conv_channel_stride=conv0.stride(1),
        conv_window_stride=conv0.stride(2),
        committed_window=committed_window,
        buffers=tuple(buffers),
    )


def _launch(
    plan: ReplaySeedPlan, buf: _BufferPlan, slots: torch.Tensor, num_slots: int, compile_only: bool
) -> None:
    block_fast = min(_BLOCK_FAST, triton.next_power_of_2(buf.num_fast))
    grid = (plan.num_layers, num_slots, triton.cdiv(buf.num_fast, block_fast))
    args = (
        buf.ptrs,
        plan.conv_ptrs,
        slots,
        buf.slot_stride,
        buf.slow_stride,
        plan.conv_slot_stride,
        plan.conv_channel_stride,
        plan.conv_window_stride,
        buf.conv_channel_offset or 0,
        buf.num_fast,
        buf.num_slow,
    )
    constexprs = dict(
        DST_DTYPE=_TORCH_TO_TL_DTYPE[buf.dtype],
        SRC_DTYPE=_TORCH_TO_TL_DTYPE[plan.conv_dtype],
        COPY=buf.conv_channel_offset is not None,
        COMMITTED=plan.committed_window,
        BLOCK_FAST=block_fast,
        BLOCK_SLOW=triton.next_power_of_2(buf.num_slow),
    )
    if compile_only:
        _seed_replay_buffer_kernel.warmup(*args, grid=grid, **constexprs)
    else:
        _seed_replay_buffer_kernel[grid](*args, **constexprs)


def seed_replay_caches(plan: ReplaySeedPlan, slots: torch.Tensor) -> None:
    """Seed the replay caches of ``slots`` (a device int64 tensor of unique slot ids).

    Equivalent, for every layer, to zeroing the slot rows of every replay buffer and
    then copying the committed conv window into the Q/K/V replay windows. Stream
    ordered, no host synchronization. Duplicate slots would race, so callers pass a
    deduplicated list.
    """
    global _seeded_admissions, _seeded_slots, _next_log_admissions
    num_slots = slots.numel()
    if num_slots == 0:
        return
    for buf in plan.buffers:
        _launch(plan, buf, slots, num_slots, compile_only=False)
    _seeded_admissions += 1
    _seeded_slots += num_slots
    if _seeded_admissions >= _next_log_admissions:
        _next_log_admissions *= 10
        logger.info(
            f"KDA replay seed: {_seeded_admissions} admissions ({_seeded_slots} slots) seeded by "
            "the batched plan"
        )


def warmup_replay_seed_kernels(plan: ReplaySeedPlan) -> None:
    """Compile the seed kernels for ``plan`` without running them.

    Admissions then never wait for the Triton JIT. No pool slot is touched: slots
    only become writable once a request has locked its pages.
    """
    slots = torch.zeros(1, dtype=torch.long, device=plan.conv_ptrs.device)
    for buf in plan.buffers:
        _launch(plan, buf, slots, 1, compile_only=True)
