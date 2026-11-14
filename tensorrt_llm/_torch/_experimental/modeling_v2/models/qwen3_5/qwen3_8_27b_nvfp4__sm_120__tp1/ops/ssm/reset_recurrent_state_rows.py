# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zero the recurrent (and optional convolution) pool rows of fresh requests.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.mamba.recurrent_state_cache.reset_recurrent_state_rows``
(tier-2 sourcing: no ``torch.ops.trtllm`` op exists). One Triton launch,
``_reset_recurrent_state_rows_kernel``; the rest of the callable is shape and
stride validation.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.mamba import recurrent_state_cache as _recurrent_state_cache

#: The per-layer pool rows: the quantized recurrent state (`mamba_ssm_cache_dtype="int8"`),
#: per value head 256 byte rows of int16 codes (two per state row) and 4 rows holding the
#: 128 fp32 row scales, and the
#: [conv_dim, d_conv - 1] = [10240, 3] convolution window.
REC_SHAPE = (48, 260, 128)
CONV_SHAPE = (10240, 3)
#: GDN layers whose same-size states the KV cache manager v2 coalesces into one slot
#: page: a layer's pool view has a slot stride of 48 layers' states.
COALESCED_LAYERS = 48


class ResetRecurrentStateRows(OpWrapper):
    """For each request `i` with `slot = state_indices[i]`: when
    `has_initial_states[i] == 0` and `0 <= slot < recurrent_states.shape[0]`,
    fill row `slot` of `recurrent_states` -- and of `conv_states`, when given --
    with +0.0. Nothing is read from either pool.

    Untouched, bitwise: every unnamed slot, every named slot whose flag is
    nonzero (the row a resumed or chunked request keeps), every negative or
    out-of-range slot (skipped, not clamped), padding between rows, the whole
    conv pool when it is `None`, both index tensors. A slot named twice is
    zeroed if any of its entries asks.

    Fusion boundary: the call decides nothing. Computing `has_initial_states`,
    assigning slots, and slicing both vectors to the context prefix are the
    caller's.

    Returns `(recurrent_states, conv_states)` -- the pools it was handed,
    written in place (`conv_states` is `None` when not passed).
    """

    ARCHS = frozenset({Arch.SM_120})

    # spec: slots (pool rows), indices (int32), flags (bool), rec_dtype, rec_shape,
    # layout "coalesced": layer 0 of a [slots, layers, ...] allocation, both pools.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="int8-typed int16-row pool [S, 48, 260, 128], coalesced: zero bytes = zero state",
            spec=dict(
                slots=16,
                indices=(3, 15, 0, 7),
                flags=(False, True, False, False),
                rec_dtype=torch.int8,
                rec_shape=REC_SHAPE,
                layout="coalesced",
            ),
        ),
    )

    note = """
    Metadata consumed: none. At the call site `state_indices` /
    `has_initial_states` are the context prefix of `Mamba2Metadata`'s int32 /
    bool buffers, and the pools are the layer's views of the cache manager v2's
    `[slots, layers, ...]` allocations, passed unsliced. Certified over the int8-typed
    int16-row pool `[S, 48, 260, 128]` as the cache manager v2 hands it (layer 0
    of 48 coalesced per slot, both pools): zero bytes are a zero int16 code and a
    zero fp32 row scale, i.e. an exact zero state; the 47 other layers between
    its rows stay bitwise. Not certified: fp32/bf16 pools, negative or
    out-of-range slots, `conv_states=None`.

    Loud, from the callable itself (ValueError): an empty recurrent pool, conv
    and recurrent slot counts differing (the in-range test uses the recurrent
    pool's), flag/index shape mismatch, non-contiguous or overlapping rows in
    either pool, any CPU tensor.

    Silent, so `is_valid` guards it: both index tensors are read as flat
    contiguous buffers of `state_indices.shape[0]` elements.

    Not guarded (data, not metadata): an index dtype too narrow for its slot
    wraps -- an int8 130 arrives as -126 and is skipped as invalid. int32 is
    what the runtime uses.

    Launch: grid `(num_requests, cdiv(max(rec_row, conv_row), 256))`, the two
    row widths masked independently; ordered on the current stream.

    Hard limits
      slot * stride(0)  int64 row base  not probed at engine slot counts
      within-row offset  int32  boundary 2^31 elements per row  not probed
        (this checkpoint's largest row is 811008)
    """

    def __call__(
        self,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        conv_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        _recurrent_state_cache.reset_recurrent_state_rows(
            recurrent_states,
            state_indices,
            has_initial_states,
            conv_states,
        )
        return recurrent_states, conv_states

    def reference(
        self,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        conv_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Both pools as the call leaves them: clone, zero the rows the rule names."""
        slots = recurrent_states.shape[0]
        rec = recurrent_states.clone()
        conv = None if conv_states is None else conv_states.clone()
        for slot, flag in zip(state_indices.tolist(), has_initial_states.tolist()):
            if flag or not 0 <= slot < slots:
                continue
            rec[slot] = 0
            if conv is not None:
                conv[slot] = 0
        return rec, conv

    def is_valid(
        self,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor,
        has_initial_states: torch.Tensor,
        conv_states: Optional[torch.Tensor] = None,
    ) -> None:
        # Measured: a stride-2 view of a 6-element index buffer reset rows
        # [0, 1] instead of [0, 1, 2] (read through the base pointer); a
        # stride-2 flag view reset row 1 instead of nothing; a [2, 3]
        # index/flag pair considered only its first 2 of 6 requests.
        assert state_indices.dim() == 1 and state_indices.is_contiguous(), (
            "state_indices must be a contiguous 1-D tensor"
        )
        assert has_initial_states.dim() == 1 and has_initial_states.is_contiguous(), (
            "has_initial_states must be a contiguous 1-D tensor"
        )


reset_recurrent_state_rows = ResetRecurrentStateRows()
