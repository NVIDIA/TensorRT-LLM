# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-step causal depthwise conv1d against a slot-indexed conv-state pool."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: qwen4_exp's GDN `[q|k|v]` width; the conv is width 4 over a 3-column pool.
QWEN4_EXP_DIM = 10240
#: Largest conv-pool slot count the cache manager was seen to allocate for
#: qwen4_exp (10 .. 461 over twelve runtime-value dumps, batch never above 32).
#: The decode step hands the whole pool over unsliced.
QWEN4_EXP_SLOTS = 461

#: Nemotron-3 Ultra's Mamba2 mixer at tep4, per rank: xBC = x 4096 | B 256 |
#: C 256 channels, width 4, with bias ...
ULTRA_TP4_DIM = 4608
#: ... read in place from the rank's in_proj output row z 4096 | xBC 4608 |
#: dt 64: a column slice with row stride 8768.
ULTRA_TP4_PROJ_WIDTH = 8768
ULTRA_TP4_XBC_OFFSET = 4096
#: MambaHybridCacheManagerV2's conv pool at that geometry (max_batch_size 32),
#: measured: 35 slots, each holding all 48 Mamba layers' `[4608, 3]` windows
#: back to back -- one layer's view is `[35, 4608, 3]`, strides
#: `(48 * 4608 * 3, 3, 1)`.
ULTRA_TP4_SLOTS = 35
ULTRA_TP4_POOL_LAYERS = 48

#: At dep4 (attention DP) every rank runs the whole mixer: xBC = x 16384 |
#: B 1024 | C 1024 channels of the in_proj row z 16384 | xBC 18432 | dt 256
#: (row stride 35072). The cache manager uses tp=1 for Mamba there; measured:
#: the same 35 slots x 48 layers, one layer's view `[35, 18432, 3]`, strides
#: `(48 * 18432 * 3, 3, 1)`.
ULTRA_DEP4_DIM = 18432
ULTRA_DEP4_PROJ_WIDTH = 35072
ULTRA_DEP4_XBC_OFFSET = 16384

#: At tep8 (the BF16 checkpoint over 8 ranks) a rank runs an eighth of the mixer: xBC =
#: x 2048 | B 128 | C 128 channels of the in_proj row z 2048 | xBC 2304 | dt 32 (row
#: stride 4384), one layer's pool view `[35, 2304, 3]`, strides `(48 * 2304 * 3, 3, 1)`.
ULTRA_TEP8_DIM = 2304
ULTRA_TEP8_PROJ_WIDTH = 4384
ULTRA_TEP8_XBC_OFFSET = 2048
#: The engine's pool at max_batch_size 256, the tep8 target's serving envelope: 271 slots.
ULTRA_TEP8_SLOTS_256 = 271


def _pinned(batch: int) -> tuple[int, ...]:
    """`batch` distinct slots of qwen4_exp's engine pool, slot 0 and the last pinned in."""
    if batch == 1:
        return (0,)
    return (0,) + tuple(i * 14 + 3 for i in range(1, batch - 1)) + (QWEN4_EXP_SLOTS - 1,)


def _ultra_pinned(batch: int) -> tuple[int, ...]:
    """`batch` distinct, scattered slots of the 35-slot Ultra pool, 0 and the last pinned in."""
    if batch == 1:
        return (0,)
    middle = tuple(1 + (i * 7) % (ULTRA_TP4_SLOTS - 2) for i in range(1, batch - 1))
    slots = (0,) + middle + (ULTRA_TP4_SLOTS - 1,)
    assert len(set(slots)) == batch
    return slots


def _ultra256_pinned(batch: int) -> tuple[int, ...]:
    """`batch` distinct, scattered slots of the 271-slot pool, 0 and the last pinned in."""
    middle = tuple(1 + (i * 7) % (ULTRA_TEP8_SLOTS_256 - 2) for i in range(1, batch - 1))
    slots = (0,) + middle + (ULTRA_TEP8_SLOTS_256 - 1,)
    assert len(set(slots)) == batch
    return slots


#: The Mamba2 decode call site: the decode rows' xBC columns of the projection,
#: written in place, bias on, one layer's view of the engine pool.
_ULTRA = dict(
    dim=ULTRA_TP4_DIM,
    bias=True,
    x_form="proj",
    proj_width=ULTRA_TP4_PROJ_WIDTH,
    proj_offset=ULTRA_TP4_XBC_OFFSET,
    pool_slots=ULTRA_TP4_SLOTS,
    pool_layers=ULTRA_TP4_POOL_LAYERS,
    pool_layer=17,
)
#: The same call at dep4 geometry.
_ULTRA_DEP4 = dict(
    _ULTRA,
    dim=ULTRA_DEP4_DIM,
    proj_width=ULTRA_DEP4_PROJ_WIDTH,
    proj_offset=ULTRA_DEP4_XBC_OFFSET,
)
#: ... and at tep8 geometry.
_ULTRA_TEP8 = dict(
    _ULTRA,
    dim=ULTRA_TEP8_DIM,
    proj_width=ULTRA_TEP8_PROJ_WIDTH,
    proj_offset=ULTRA_TEP8_XBC_OFFSET,
)


class CausalConv1dUpdate(OpWrapper):
    """One causal depthwise conv step per batch row against that row's cached
    window, then the window's advance -- `x` and the pool both written in place.

    Row `b`, slot `s = conv_state_indices[b]` (or `b`), new tokens `i < seqlen`:
    `history = [conv_state[s, c, :], x[b, c, :]]`, the window is its last
    `W - 1 + seqlen` columns, `x[b, c, i] = act(bias[c] + sum_j weight[c, j] *
    window[i + j])` in fp32 rounded once, and `conv_state[s] = history[-state_len:]`
    with `state_len = conv_state.shape[2]`. With `cache_seqlens` the pool is a
    ring instead: position `p` lives at `p % state_len`, `cache_seqlens[b]` is
    the first new token's absolute position, and nothing shifts.

    Fusion boundary: window read, convolution, bias, SiLU and state write are
    inside. The projection producing `x`, slot allocation and the q/k/v or
    x/B/C split are the caller's.

    Returns `x` -- the same object, written in place. A 2-D `[batch, dim]` `x`
    is handed to the op as an `unsqueeze(-1)` view (metadata only; the op
    writes through it). `conv_state` is advanced in place and not returned.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103, Arch.SM_107})

    # spec keys: dim, width (4), state_len (3), seqlen (1), batch, pool_slots,
    # pool_layers/pool_layer (the pool is layer `pool_layer`'s view of a
    # `[slots, pool_layers, dim, state_len]` buffer), slots ("random" | tuple |
    # None), dtype, bias, silu, cache_seqlens, pad_slot_id, x_form ("3d" |
    # "2d" | "2d_row_slice" | "token_major" | "proj": 2-D x the columns
    # [proj_offset, proj_offset + dim) of rows [rows_ahead, rows_ahead + batch)
    # of a token-major `[rows_ahead + batch, proj_width]` buffer),
    # strided_state, padding_rows (the last rows name one shared slot, as a
    # CUDA-graph-padded batch does; their outputs and that slot are left out
    # of the comparison).
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"qwen4_exp GDN decode batch {batch}: 2-D x, 461-slot pool",
                spec=dict(
                    dim=QWEN4_EXP_DIM,
                    batch=batch,
                    pool_slots=QWEN4_EXP_SLOTS,
                    slots=_pinned(batch),
                    x_form="2d",
                ),
            )
            for batch in (1, 2, 17, 32)
        )
        + (
            Cell(
                why="qwen4_exp mixed batch: 2-D x the decode-row slice of the projection",
                spec=dict(
                    dim=QWEN4_EXP_DIM,
                    batch=8,
                    pool_slots=QWEN4_EXP_SLOTS,
                    slots=_pinned(8),
                    x_form="2d_row_slice",
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"Mamba2 tep4 decode batch {batch}: xBC slice of the projection, engine pool",
                spec=dict(_ULTRA, batch=batch, slots=_ultra_pinned(batch)),
            )
            for batch in (1, 2, 8, 17, 32)
        )
        + (
            Cell(
                why="Mamba2 tep4 mixed batch: decode rows below 37 prefill rows",
                spec=dict(_ULTRA, batch=8, rows_ahead=37, slots=_ultra_pinned(8)),
            ),
            Cell(
                why="Mamba2 tep4 graph-padded batch: 3 padding rows share one dummy slot",
                spec=dict(
                    _ULTRA,
                    batch=8,
                    slots=(0, 9, 23, 5, 16) + (ULTRA_TP4_SLOTS - 1,) * 3,
                    padding_rows=3,
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"Mamba2 dep4 decode batch {batch}: xBC slice of the projection, engine pool",
                spec=dict(_ULTRA_DEP4, batch=batch, slots=_ultra_pinned(batch)),
            )
            for batch in (1, 7, 32)
        )
        + (
            Cell(
                why="Mamba2 dep4 mixed batch: decode rows below 37 prefill rows",
                spec=dict(_ULTRA_DEP4, batch=8, rows_ahead=37, slots=_ultra_pinned(8)),
            ),
            Cell(
                why="Mamba2 dep4 graph-padded batch: 3 padding rows share one dummy slot",
                spec=dict(
                    _ULTRA_DEP4,
                    batch=8,
                    slots=(0, 9, 23, 5, 16) + (ULTRA_TP4_SLOTS - 1,) * 3,
                    padding_rows=3,
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"Mamba2 tep8 decode batch {batch}: xBC slice of the projection, engine pool",
                spec=dict(_ULTRA_TEP8, batch=batch, slots=_ultra_pinned(batch)),
            )
            for batch in (1, 2, 8, 17, 32)
        )
        + (
            Cell(
                why="Mamba2 tep8 mixed batch: decode rows below 37 prefill rows",
                spec=dict(_ULTRA_TEP8, batch=8, rows_ahead=37, slots=_ultra_pinned(8)),
            ),
            Cell(
                why="Mamba2 tep8 decode batch 256, the serving envelope's ceiling: the engine's "
                "271-slot pool",
                spec=dict(
                    _ULTRA_TEP8,
                    batch=256,
                    pool_slots=ULTRA_TEP8_SLOTS_256,
                    slots=_ultra256_pinned(256),
                ),
            ),
            Cell(
                why="Mamba2 tep8 graph-padded batch: 3 padding rows share one dummy slot",
                spec=dict(
                    _ULTRA_TEP8,
                    batch=8,
                    slots=(0, 9, 23, 5, 16) + (ULTRA_TP4_SLOTS - 1,) * 3,
                    padding_rows=3,
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"decode batch {batch}, [batch, dim, 1] x, 128-slot pool",
                spec=dict(dim=QWEN4_EXP_DIM, batch=batch, pool_slots=128, slots="random"),
            )
            for batch in (1, 2, 17, 64)
        )
        + (
            Cell(
                why="pad_slot_id -1 rows: x rows and the pool untouched",
                spec=dict(dim=QWEN4_EXP_DIM, batch=6, pool_slots=128, slots=(3, -1, 7, 11, -1, 2)),
            ),
            Cell(
                why="pad_slot_id 9: the sentinel is compared by value",
                spec=dict(
                    dim=QWEN4_EXP_DIM, batch=4, pool_slots=128, slots=(5, 9, 0, 9), pad_slot_id=9
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"bias={bias}, silu={silu} (general kernel unless silu)",
                spec=dict(
                    dim=QWEN4_EXP_DIM,
                    batch=4,
                    pool_slots=128,
                    slots="random",
                    bias=bias,
                    silu=silu,
                ),
            )
            for bias in (False, True)
            for silu in (True, False)
        )
        + tuple(
            Cell(
                why=f"state_len {state_len} > W-1: the whole state shifts left",
                spec=dict(dim=512, batch=3, pool_slots=16, slots="random", state_len=state_len),
            )
            for state_len in (4, 5, 8)
        )
        + tuple(
            Cell(
                why=f"multi-token step: seqlen {seqlen}, state_len {state_len}",
                spec=dict(
                    dim=512,
                    batch=3,
                    pool_slots=16,
                    slots="random",
                    seqlen=seqlen,
                    state_len=state_len,
                ),
            )
            for seqlen in (2, 3, 5)
            for state_len in (3, 5)
        )
        + tuple(
            Cell(
                why=f"circular ring, state_len {state_len}, seqlen {seqlen}, a pad row",
                spec=dict(
                    dim=512,
                    batch=4,
                    pool_slots=16,
                    slots=(3, 7, -1, 11),
                    seqlen=seqlen,
                    state_len=state_len,
                    cache_seqlens=(0, 1, state_len - 1, 37),
                ),
            )
            for state_len in (4, 8)
            for seqlen in (1, 3)
        )
        + tuple(
            Cell(
                why=f"{str(dtype).removeprefix('torch.')} with bias",
                spec=dict(dim=512, batch=4, pool_slots=16, slots="random", dtype=dtype, bias=True),
            )
            for dtype in (torch.bfloat16, torch.float16, torch.float32)
        )
        + (
            Cell(
                why="conv_state_indices None: row b uses slot b, batch-sized pool",
                spec=dict(dim=512, batch=4, pool_slots=4, slots=None),
            ),
            Cell(
                why="x a [batch, tokens, dim].transpose(1, 2) view",
                spec=dict(
                    dim=512, batch=3, seqlen=4, pool_slots=16, slots=(2, 5, 1), x_form="token_major"
                ),
            ),
            Cell(
                why="conv_state a [:, :, ::2] view: interleaved columns untouched",
                spec=dict(dim=512, batch=3, pool_slots=16, slots=(2, 5, 1), strided_state=True),
            ),
        )
        + tuple(
            Cell(
                why=f"dim {dim}, off the 128-channel block",
                spec=dict(dim=dim, batch=2, pool_slots=8, slots="random"),
            )
            for dim in (1, 127, 128, 129, 513)
        )
    )

    note = """
    Metadata consumed: none.

    Call sites. Both run the dedicated seqlen-1 kernel: silu, no ring,
    `state_len == W - 1`; every other cell runs the general one.
    qwen4_exp, the GDN decode step: 2-D `[num_decodes, 10240]` x (the decode
    rows of the projection), the layer's whole conv pool `[slots, 10240, 3]`,
    W 4, no bias, SiLU on, `cache_seqlens` None, int32 slots, `pad_slot_id`
    -1. Certified at 128 and 461 pool slots; at 461 the result is bitwise
    identical to 128 for the same slot content.
    nemotron_h Ultra tep4, the Mamba2 decode step: 2-D
    `x = proj[T_ctx:, 4096:8704]` -- the decode rows' xBC columns of the
    rank's in_proj output (row stride 8768), activated in place, so x/B/C are
    read back as column views of the same buffer -- W 4 with bias, SiLU on,
    `conv_state` the layer's engine view `[35, 4608, 3]` with slot stride
    48 * 4608 * 3, `conv_state_indices` the decode rows' slice of the step's
    Mamba2 state indices, `pad_slot_id` -1. The `Mamba2 tep4` cells are that
    call; the projection's z and dt columns, the rows above the decode rows and
    the other 47 layers' windows are gated bitwise unchanged.
    nemotron_h Ultra dep4, the same call unsharded: `x = proj[T_ctx:,
    16384:34816]` (row stride 35072), `conv_state` the layer's view
    `[35, 18432, 3]`, slot stride 48 * 18432 * 3 -- the `Mamba2 dep4` cells.

    Everything metadata the op checks itself, loudly: width 2..4, `state_len
    >= W - 1`, one dtype across x/conv_state/weight (bias matches weight),
    weight shape, int32 contiguous length-batch `conv_state_indices`, int32
    `cache_seqlens`, a batch-sized pool when indices are absent. So no guard.

    Not guarded because it is content, not metadata: two rows naming one slot
    race -- the slot ended up matching neither row's expected state in 10 of
    10 runs while both outputs matched. Slot uniqueness is the caller's, with
    one exception the engine makes on purpose: a CUDA-graph-padded decode
    batch repeats a single dummy request, so all its padding rows name that
    request's slot. The slot's content and those rows' outputs are then
    whatever the race left -- both discarded -- while the real rows and every
    other slot are exact; the graph-padded cell gates exactly that.
    Out-of-range slot values are an illegal access, not probed.

    A row whose index equals `pad_slot_id` is skipped: its x rows and the pool
    stay bitwise unchanged (linear and ring mode). Any value works as the
    sentinel, including one that collides with a live slot.

    Hard limits
      slot * slot stride  uint32 offset  boundary 2^32 elements; the Ultra
        pool view reaches 2.3e7.
    """

    def __call__(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        silu_activation: bool,
        cache_seqlens: Optional[torch.Tensor],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
    ) -> torch.Tensor:
        torch.ops.trtllm.causal_conv1d_update(
            x.unsqueeze(-1) if x.dim() == 2 else x,
            conv_state,
            weight,
            bias,
            silu_activation,
            cache_seqlens,
            conv_state_indices,
            pad_slot_id,
        )
        return x

    def reference(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        silu_activation: bool,
        cache_seqlens: Optional[torch.Tensor],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x, conv_state)` as the call leaves them; mutates nothing."""
        x3 = x.unsqueeze(-1) if x.dim() == 2 else x
        batch, dim, seqlen = x3.shape
        width = weight.shape[1]
        state_len = conv_state.shape[2]
        out = x3.clone()
        new_state = conv_state.clone()
        slots = None if conv_state_indices is None else conv_state_indices.tolist()
        starts = None if cache_seqlens is None else cache_seqlens.tolist()
        for b in range(batch):
            slot = b if slots is None else slots[b]
            if slots is not None and slot == pad_slot_id:
                continue
            state = conv_state[slot].float()
            tokens = x3[b].float()
            if starts is None:
                history = torch.cat([state, tokens], dim=1)
                window = history[:, state_len - (width - 1) :]
                new_state[slot] = history[:, -state_len:].to(conv_state.dtype)
            else:
                start = starts[b] % state_len
                read = [(start - (width - 1) + i) % state_len for i in range(width - 1)]
                window = torch.cat([state[:, read], tokens], dim=1)
                advanced = state.clone()
                for i in range(seqlen):
                    advanced[:, (start + i) % state_len] = tokens[:, i]
                new_state[slot] = advanced.to(conv_state.dtype)
            acc = torch.zeros(dim, seqlen, dtype=torch.float32, device=x.device)
            for j in range(width):
                acc = acc + weight[:, j].float().unsqueeze(1) * window[:, j : j + seqlen]
            if bias is not None:
                acc = acc + bias.float().unsqueeze(1)
            if silu_activation:
                acc = acc * torch.sigmoid(acc)
            out[b] = acc.to(x.dtype)
        return (out.squeeze(-1) if x.dim() == 2 else out), new_state


causal_conv1d_update = CausalConv1dUpdate()
