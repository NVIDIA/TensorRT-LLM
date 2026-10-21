# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-step causal depthwise conv1d against a slot-indexed conv-state pool."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's GDN `[q|k|v]` width; the conv is width 4 over a 3-column pool.
CKPT_DIM = 10240
#: Largest conv-pool slot count the cache manager was seen to allocate for this
#: checkpoint (10 .. 461 over twelve runtime-value dumps, batch never above 32).
#: The decode step hands the whole pool over unsliced.
ENGINE_SLOTS = 461


def _pinned(batch: int) -> tuple[int, ...]:
    """`batch` distinct slots of the engine pool, slot 0 and the last pinned in."""
    if batch == 1:
        return (0,)
    return (0,) + tuple(i * 14 + 3 for i in range(1, batch - 1)) + (ENGINE_SLOTS - 1,)


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
    inside. The projection producing `x`, slot allocation and the q/k/v split
    are the caller's.

    Returns `x` -- the same object, written in place. A 2-D `[batch, dim]` `x`
    is handed to the op as an `unsqueeze(-1)` view (metadata only; the op
    writes through it). `conv_state` is advanced in place and not returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec keys: dim, width (4), state_len (3), seqlen (1), batch, pool_slots,
    # slots ("random" | tuple | None), dtype, bias, silu, cache_seqlens,
    # pad_slot_id, x_form ("3d" | "2d" | "2d_row_slice" | "token_major"),
    # strided_state.
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"decode batch {batch}, call-site 2-D x, 461-slot pool",
                spec=dict(
                    dim=CKPT_DIM,
                    batch=batch,
                    pool_slots=ENGINE_SLOTS,
                    slots=_pinned(batch),
                    x_form="2d",
                ),
            )
            for batch in (1, 2, 17, 32)
        )
        + (
            Cell(
                why="mixed batch: 2-D x is the decode-row slice of the projection",
                spec=dict(
                    dim=CKPT_DIM,
                    batch=8,
                    pool_slots=ENGINE_SLOTS,
                    slots=_pinned(8),
                    x_form="2d_row_slice",
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"decode batch {batch}, [batch, dim, 1] x, 128-slot pool",
                spec=dict(dim=CKPT_DIM, batch=batch, pool_slots=128, slots="random"),
            )
            for batch in (1, 2, 17, 64)
        )
        + (
            Cell(
                why="pad_slot_id -1 rows: x rows and the pool untouched",
                spec=dict(dim=CKPT_DIM, batch=6, pool_slots=128, slots=(3, -1, 7, 11, -1, 2)),
            ),
            Cell(
                why="pad_slot_id 9: the sentinel is compared by value",
                spec=dict(dim=CKPT_DIM, batch=4, pool_slots=128, slots=(5, 9, 0, 9), pad_slot_id=9),
            ),
        )
        + tuple(
            Cell(
                why=f"bias={bias}, silu={silu} (general kernel unless silu)",
                spec=dict(
                    dim=CKPT_DIM, batch=4, pool_slots=128, slots="random", bias=bias, silu=silu
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

    Call site: the GDN decode step -- 2-D `[num_decodes, 10240]` x (the decode
    rows of the projection), the layer's whole conv pool `[slots, 10240, 3]`,
    W 4, no bias, SiLU on, `cache_seqlens` None, int32 slots, `pad_slot_id` -1.
    That exact shape (silu, no ring, `state_len == W - 1`) runs a dedicated
    seqlen-1 kernel; every other cell runs the general one. Certified at 128
    and 461 pool slots; at 461 the result is bitwise identical to 128 for the
    same slot content.

    Everything metadata the op checks itself, loudly: width 2..4, `state_len
    >= W - 1`, one dtype across x/conv_state/weight (bias matches weight),
    weight shape, int32 contiguous length-batch `conv_state_indices`, int32
    `cache_seqlens`, a batch-sized pool when indices are absent. So no guard.

    Not guarded because it is content, not metadata: two rows naming one slot
    race -- the slot ended up matching neither row's expected state in 10 of
    10 runs while both outputs matched. Slot uniqueness is the caller's.
    Out-of-range slot values are an illegal access, not probed.

    A row whose index equals `pad_slot_id` is skipped: its x rows and the pool
    stay bitwise unchanged (linear and ring mode). Any value works as the
    sentinel, including one that collides with a live slot.

    Hard limits
      none probed.
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
