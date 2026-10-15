# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-token causal depthwise conv1d step (Triton), with optional per-step
window snapshots for speculative verification.

Wraps ``tensorrt_llm._torch.modules.mamba.causal_conv1d_triton.causal_conv1d_update``
-- one Triton launch (``_causal_conv1d_update_kernel``) after allocating the
output. There is no torch-op boundary around this kernel. It is *not*
``torch.ops.trtllm.causal_conv1d_update``: that op writes `x` in place, has a
ring mode, takes `state_len` from the pool, has no snapshots, and multiplies in
fp32 -- the two are not substitutable.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import (
    causal_conv1d_update as _causal_conv1d_triton_update,
)

#: This checkpoint's GDN `[q|k|v]` width; the conv is width 4 over a 3-column pool.
CKPT_DIM = 10240
#: Largest conv-pool slot count the cache manager was seen to allocate for this
#: checkpoint (10 .. 461 over twelve runtime-value dumps). The verify path hands
#: the whole pool over unsliced.
ENGINE_SLOTS = 461
#: The engine's `max_num_sequences`: the snapshot buffer's row count.
ENGINE_SNAPSHOT_ROWS = 32


def _pinned(batch: int) -> tuple[int, ...]:
    """`batch` distinct slots of the engine pool, slot 0 and the last pinned in."""
    if batch == 1:
        return (0,)
    return (0,) + tuple(i * 14 + 3 for i in range(1, batch - 1)) + (ENGINE_SLOTS - 1,)


class CausalConv1dTritonUpdate(OpWrapper):
    """Convolve each row's new tokens against its cached window, out of place,
    and advance the window; optionally snapshot the window after every token.

    Row `b`, slot `s = conv_state_indices[b]` (or `b`), `off = 0` (or
    `num_accepted_tokens[b] - 1`): `window = [conv_state[s, :, off : off+W-1],
    x[b]]`, `out[b, c, i] = act(bias[c] + sum_j fp32(weight[c, j] * window[i+j]))`
    -- each product rounded to the operand dtype, accumulated in fp32, SiLU as
    `acc / (1 + exp(-acc))`, rounded once. The state is then rewritten from
    column 0: the last `W - 1` window columns (or, with `num_accepted_tokens`,
    `W - 1 + seqlen - 1` columns rebased past the offset). Columns beyond that
    are untouched -- `state_len` comes from `W`, never from the pool.

    With `intermediate_conv_window`, snapshot `i` of row `b` --
    `[intermediate_state_indices[b], i, :, :W-1]` -- is the window after token
    `i`; after acceptance of `k` tokens the cache is snapshot `k - 1`.

    Fusion boundary: window read, conv, bias, SiLU, state write and snapshots
    are inside. The projection, the q/k/v split and the post-acceptance
    promotion of a snapshot into the pool are the caller's.

    Returns a fresh tensor with `x`'s shape (2-D in, 2-D out); `x` is read-only.
    A row whose slot equals `pad_slot_id` is not written at all -- its result
    rows are allocator garbage -- and its pool slot and snapshot rows stay
    bitwise unchanged. `conv_state` and `intermediate_conv_window` are written
    in place and not returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec keys: dim, width (4), state_len (pool columns, 3), seqlen (2), batch,
    # pool_slots, slots ("random" | tuple | None), dtype, bias, activation
    # ("silu"), pad_slot_id, x_form ("token_major" | "contiguous" | "2d"),
    # snapshots (rows of the snapshot buffer, or 0 for none), accepted,
    # launch_dependent_kernels.
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(
                why=f"verify at the 461-slot pool, batch {batch}, {seqlen} tokens/row",
                spec=dict(
                    dim=CKPT_DIM,
                    batch=batch,
                    seqlen=seqlen,
                    pool_slots=ENGINE_SLOTS,
                    slots=_pinned(batch),
                    snapshots=ENGINE_SNAPSHOT_ROWS,
                ),
            )
            for batch, seqlen in ((1, 2), (8, 3), (32, 2), (17, 4))
        )
        + tuple(
            Cell(
                why=f"verify, 128-slot pool, batch {batch}, {seqlen} tokens/row",
                spec=dict(
                    dim=CKPT_DIM,
                    batch=batch,
                    seqlen=seqlen,
                    pool_slots=128,
                    slots="random",
                    snapshots=batch,
                ),
            )
            for batch, seqlen in ((1, 2), (4, 2), (8, 3), (17, 4), (64, 2))
        )
        + (
            Cell(
                why="pad_slot_id -1 rows: pool slot and snapshot rows untouched",
                spec=dict(
                    dim=512, batch=6, pool_slots=16, slots=(3, -1, 7, 11, -1, 2), snapshots=6
                ),
            ),
            Cell(
                why="pad_slot_id 9: the sentinel is compared by value",
                spec=dict(dim=512, batch=4, pool_slots=16, slots=(5, 9, 0, 9), pad_slot_id=9),
            ),
            Cell(
                why="conv_state_indices None: row b reads and writes slot b",
                spec=dict(dim=512, batch=4, pool_slots=4, slots=None),
            ),
        )
        + tuple(
            Cell(
                why=f"activation={activation!r}",
                spec=dict(dim=512, batch=4, pool_slots=16, slots="random", activation=activation),
            )
            for activation in (None, False, True, "swish")
        )
        + tuple(
            Cell(
                why=f"bias, activation={activation!r}: added in the fp32 accumulator",
                spec=dict(
                    dim=512,
                    batch=4,
                    pool_slots=16,
                    slots="random",
                    bias=True,
                    activation=activation,
                ),
            )
            for activation in ("silu", None)
        )
        + (
            Cell(
                why="2-D [batch, dim] one-token step returns 2-D",
                spec=dict(dim=512, batch=12, seqlen=1, pool_slots=16, slots="random", x_form="2d"),
            ),
        )
        + tuple(
            Cell(
                why=f"pool {state_len} columns wide: columns past W-1 are inert",
                spec=dict(dim=512, batch=3, pool_slots=16, slots="random", state_len=state_len),
            )
            for state_len in (4, 6)
        )
        + (
            Cell(
                why="num_accepted_tokens 1/2/3/1 rolls the window to column 0",
                spec=dict(
                    dim=512,
                    batch=4,
                    seqlen=3,
                    state_len=5,  # W-1 + (seqlen-1): the pool this form needs
                    pool_slots=16,
                    slots="random",
                    accepted=(1, 2, 3, 1),
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"conv width {width}",
                spec=dict(
                    dim=512,
                    batch=3,
                    width=width,
                    state_len=width - 1,
                    pool_slots=16,
                    slots="random",
                ),
            )
            for width in (2, 3)
        )
        + tuple(
            Cell(
                why=f"dim {dim}, off the 128-channel block",
                spec=dict(dim=dim, batch=2, pool_slots=8, slots="random"),
            )
            for dim in (1, 127, 128, 129, 513)
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
                why="x contiguous [batch, dim, seqlen] rather than token-major",
                spec=dict(
                    dim=512, batch=3, seqlen=4, pool_slots=16, slots=(2, 5, 1), x_form="contiguous"
                ),
            ),
            Cell(
                why="launch_dependent_kernels=True (PDL trigger)",
                spec=dict(
                    dim=512, batch=4, pool_slots=16, slots="random", launch_dependent_kernels=True
                ),
            ),
        )
    )

    note = """
    Metadata consumed: none.

    Call site: the GDN speculative verify step -- `x` the `[n, 10240, p]`
    transposed view of `[n, p, 10240]` (p = runtime_draft_len + 1), the layer's
    whole conv pool `[slots, 10240, 3]`, W 4, no bias, `"silu"`, the verify rows'
    slots, `pad_slot_id` -1, snapshots into `[max_num_sequences, p, 10240, 3]`
    with `intermediate_state_indices = arange(n)`, `num_accepted_tokens` None.
    The op advances the pool by all `p` tokens whatever is accepted later; the
    snapshots are what a rollback has to read. Certified at 128 and 461 pool
    slots (bitwise identical for the same slot content), snapshot buffers of
    `batch` and of 32 rows.

    Raise loudly, so not guarded: width 1 (triton CompilationError); a missing
    `intermediate_state_indices` with snapshots and slot indices given.

    Not guarded (content): duplicate slots or duplicate snapshot rows race;
    `num_accepted_tokens[b]` must lie in `[1, seqlen]` (certified 1..3 at
    seqlen 3). Out-of-range slot values: illegal access, not probed.

    Accepted and inert: `launch_dependent_kernels` (bitwise identical either
    way). Accepted but not exposed: `cache_seqlens` (ignored by the kernel --
    there is no ring mode), the eagle-tree mask trio, `validate_data`,
    `metadata`, the `_block_n`/`_num_warps`/`_num_stages` overrides. A
    `conv_state` dtype different from `x` is accepted and uncertified.

    Hard limits
      none probed.
    """

    def __call__(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: Union[bool, str, None],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        intermediate_conv_window: Optional[torch.Tensor] = None,
        intermediate_state_indices: Optional[torch.Tensor] = None,
        launch_dependent_kernels: bool = False,
    ) -> torch.Tensor:
        return _causal_conv1d_triton_update(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            intermediate_conv_window=intermediate_conv_window,
            intermediate_state_indices=intermediate_state_indices,
            pad_slot_id=pad_slot_id,
            launch_dependent_kernels=launch_dependent_kernels,
        )

    def reference(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: Union[bool, str, None],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        intermediate_conv_window: Optional[torch.Tensor] = None,
        intermediate_state_indices: Optional[torch.Tensor] = None,
        launch_dependent_kernels: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """`(out, conv_state, intermediate_conv_window)` as the call leaves
        them; mutates nothing. `launch_dependent_kernels` is ignored.

        Rows the op leaves unwritten (padded) are NaN in `out`; the test
        skips them. Products are formed in the operand dtype, as the kernel
        forms them -- multiplying in fp32 instead misses the default band on
        about a third of the elements.
        """
        x3 = x.unsqueeze(-1) if x.dim() == 2 else x
        batch, dim, seqlen = x3.shape
        width = weight.shape[1]
        silu = activation in (True, "silu", "swish")
        state_len = width - 1 + (seqlen - 1 if num_accepted_tokens is not None else 0)
        out = torch.full(x3.shape, float("nan"), dtype=x.dtype, device=x.device)
        new_state = conv_state.clone()
        inter = None if intermediate_conv_window is None else intermediate_conv_window.clone()
        slots = None if conv_state_indices is None else conv_state_indices.tolist()
        accepted = None if num_accepted_tokens is None else num_accepted_tokens.tolist()
        rows = None if intermediate_state_indices is None else intermediate_state_indices.tolist()
        for b in range(batch):
            slot = b if slots is None else slots[b]
            if slots is not None and slot == pad_slot_id:
                continue
            off = 0 if accepted is None else accepted[b] - 1
            state = conv_state[slot]
            full = torch.cat([state[:, off : off + width - 1], x3[b]], dim=1)
            acc = torch.zeros(dim, seqlen, dtype=torch.float32, device=x.device)
            for j in range(width):
                acc = acc + (weight[:, j].unsqueeze(1) * full[:, j : j + seqlen]).float()
            if bias is not None:
                acc = acc + bias.float().unsqueeze(1)
            if silu:
                acc = acc / (1.0 + torch.exp(-acc))
            out[b] = acc.to(x.dtype)
            if inter is not None:
                row = b if rows is None else rows[b]
                for i in range(seqlen):
                    inter[row, i, :dim, : width - 1] = full[:, i + 1 : i + width]
            if accepted is None:
                new_state[slot, :, : width - 1] = full[:, seqlen : seqlen + width - 1]
            else:
                new_state[slot, :, :state_len] = torch.cat(
                    [state[:, off + 1 : off + width - 1], x3[b]], dim=1
                )
        return (out.squeeze(-1) if x.dim() == 2 else out), new_state, inter

    def is_valid(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: Union[bool, str, None],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        intermediate_conv_window: Optional[torch.Tensor] = None,
        intermediate_state_indices: Optional[torch.Tensor] = None,
        launch_dependent_kernels: bool = False,
    ) -> None:
        dim = x.shape[1]
        seqlen = 1 if x.dim() == 2 else x.shape[2]
        width = weight.shape[1]
        state_len = width - 1 + (seqlen - 1 if num_accepted_tokens is not None else 0)
        # The tap loop is specialised for W 2..4: W 5 runs and returned an
        # answer up to 47.2 off (mean 1.89). W 1 fails to compile.
        assert 2 <= width <= 4, (
            f"weight width must be 2, 3 or 4; got {width} -- width 5 is accepted "
            "by the kernel and silently computes a wrong answer"
        )
        # Reads and writes `state_len` columns unmasked against the pool: a
        # 2-column pool at W 4 wrote one element into the next slot.
        assert conv_state.shape[2] >= state_len, (
            f"conv_state must hold at least {state_len} columns (width={width}, "
            f"seqlen={seqlen}, num_accepted_tokens="
            f"{'given' if num_accepted_tokens is not None else 'None'}); got "
            f"{conv_state.shape[2]} -- a shorter pool is accepted and writes past "
            "the slot into the next one"
        )
        # The channel axis is masked against x's dim only: a half-width pool
        # changed 384 elements of unnamed slots; a half-height weight was read
        # past its end.
        assert conv_state.shape[1] >= dim and weight.shape[0] >= dim, (
            f"conv_state and weight must cover all {dim} channels of x; got "
            f"{conv_state.shape[1]} and {weight.shape[0]} -- a narrower pool or "
            "weight is accepted and reads/writes out of bounds"
        )
        if intermediate_conv_window is not None:
            # No bound on the step axis: a 2-step buffer at seqlen 4 wrote
            # into the next row.
            assert (
                intermediate_conv_window.shape[1] >= seqlen
                and intermediate_conv_window.shape[2] >= dim
                and intermediate_conv_window.shape[3] >= width - 1
            ), (
                f"intermediate_conv_window must be at least [rows, {seqlen}, {dim}, "
                f"{width - 1}]; got {tuple(intermediate_conv_window.shape)} -- an "
                "undersized buffer is accepted and writes past the row it was given"
            )


causal_conv1d_triton_update = CausalConv1dTritonUpdate()
