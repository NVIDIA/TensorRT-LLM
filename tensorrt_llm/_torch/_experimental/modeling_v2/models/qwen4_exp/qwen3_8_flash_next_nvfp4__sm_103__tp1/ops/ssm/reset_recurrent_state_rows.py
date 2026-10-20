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

#: This checkpoint's per-layer pool rows: a [48, 128, 128] recurrent state and a
#: [conv_dim, d_conv - 1] = [10240, 3] convolution window.
REC_SHAPE = (48, 128, 128)
CONV_SHAPE = (10240, 3)
#: Largest slot count the cache manager was seen to allocate both pools at for
#: this checkpoint (10 .. 461 over twelve runtime-value dumps, batch never above
#: 32). The target hands both pools over unsliced.
ENGINE_SLOTS = 461


def _engine_case(num_seqs: int, flip: bool) -> dict:
    """`num_seqs` distinct slots over the engine pool, slot 0 and 460 live;
    `flip` moves which of the two is the one reset."""
    slots = (0, ENGINE_SLOTS - 1) + tuple(i * 13 + 4 for i in range(num_seqs - 2))
    flags = tuple((i % 2 == 1) != flip for i in range(num_seqs))
    return dict(slots=ENGINE_SLOTS, indices=slots, flags=flags)


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

    ARCHS = frozenset({Arch.SM_103})

    # spec: slots (pool rows), indices, flags, index_dtype (int32), flag_dtype
    # (bool), rec_dtype (fp32), conv (True), rec_shape/conv_shape (checkpoint),
    # layout: "plain" | "row_padding" (stride(0) > row) | "layer_view"
    # (pool[1] of a [3, slots, ...] allocation); `indices="perm"` is a random
    # permutation of every slot with random flags.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="checkpoint geometry prefill batch, mixed flags",
            spec=dict(slots=6, indices=(3, 0, 5, 1), flags=(False, True, False, True)),
        ),
        Cell(
            why="engine pool 461, 4 requests, slot 460 reset",
            spec=_engine_case(4, flip=False),
        ),
        Cell(
            why="engine pool 461, 32 requests, slot 460 reset",
            spec=_engine_case(32, flip=False),
        ),
        Cell(
            why="engine pool 461, 32 requests, slot 0 reset",
            spec=_engine_case(32, flip=True),
        ),
        Cell(
            why="every request continues: pools bitwise unchanged",
            spec=dict(slots=6, indices=(0, 1, 2, 3, 4, 5), flags=(True,) * 6),
        ),
        Cell(
            why="one named slot; the other five untouched",
            spec=dict(slots=6, indices=(2,), flags=(False,)),
        ),
    )
    CELLS += tuple(
        Cell(
            why=f"negative and out-of-range slots skipped, {str(d)[6:]}",
            spec=dict(slots=6, indices=(-1, 6, 23, -99999, 2), flags=(False,) * 5, index_dtype=d),
        )
        for d in (torch.int32, torch.int64)
    )
    CELLS += tuple(
        Cell(
            why=f"index dtype {str(d)[6:]} widened after the load",
            spec=dict(slots=4, indices=(1,), flags=(False,), index_dtype=d),
        )
        for d in (torch.int8, torch.int16, torch.int64, torch.uint8)
    )
    CELLS += tuple(
        Cell(
            why=f"flag dtype {str(d)[6:]}: a nonzero test, not a low bit",
            spec=dict(
                slots=4,
                indices=(0, 1, 2, 3),
                flags=(0, 1, 1, 1) if d is torch.bool else (0, 1, 2, 3),
                flag_dtype=d,
            ),
        )
        for d in (torch.bool, torch.uint8, torch.int32, torch.int64, torch.float32)
    )
    CELLS += (
        Cell(
            why="conv_states omitted: that pool is not touched",
            spec=dict(slots=6, indices=(4,), flags=(False,), conv=False),
        ),
    )
    CELLS += tuple(
        Cell(
            why=f"slot named twice, flags {flags}",
            spec=dict(slots=6, indices=(2, 2), flags=flags),
        )
        for flags in ((False, True), (True, False), (True, True))
    )
    CELLS += (
        Cell(
            why="stride(0) > row size: the padding lane untouched",
            spec=dict(slots=6, indices=(1,), flags=(False,), conv=False, layout="row_padding"),
        ),
        Cell(
            why="pool[layer] of a [layers, slots, ...] allocation",
            spec=dict(slots=4, indices=(2, 0), flags=(False, True), layout="layer_view"),
        ),
        Cell(
            why="empty batch launches nothing",
            spec=dict(slots=6, indices=(), flags=()),
        ),
    )
    CELLS += tuple(
        Cell(
            why=f"recurrent pool {str(d)[6:]} (--mamba_ssm_cache_dtype)",
            spec=dict(slots=3, indices=(1,), flags=(False,), rec_dtype=d),
        )
        for d in (torch.bfloat16, torch.float16)
    )
    CELLS += (
        Cell(
            why="conv row wider than the recurrent row",
            spec=dict(slots=3, indices=(1,), flags=(False,), rec_shape=(8,), conv_shape=(5000,)),
        ),
        Cell(
            why="64-slot random permutation: slot-indexed, not position-indexed",
            spec=dict(slots=64, indices="perm", flags="random"),
        ),
    )

    note = """
    Metadata consumed: none. At the call site `state_indices` /
    `has_initial_states` are the context prefix of `Mamba2Metadata`'s int32 /
    bool buffers, and the pools are the layer's views of the cache manager's
    `[layers, slots, ...]` allocations, passed unsliced. Certified at 3..64
    and 461 slots; at 461 the result is bitwise identical to the 64-slot
    result for the same slot content.

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
      slot * stride(0)  int64 row base  probed at 460 * 786432 = 3.6e8  exact
      within-row offset  int32  boundary 2^31 elements per row  not probed
        (this checkpoint's largest row is 786432)
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
