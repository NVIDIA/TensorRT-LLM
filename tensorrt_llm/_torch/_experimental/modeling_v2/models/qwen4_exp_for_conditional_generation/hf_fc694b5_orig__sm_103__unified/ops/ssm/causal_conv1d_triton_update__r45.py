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
#: A conv-pool slot count of a full-size engine run. The kernel addresses a slot by its
#: index alone, so a pool of another size (a bf16 recurrent pool holds more slots in the
#: same memory, and the conv pool shares its slot space) takes the same path. The verify
#: path hands the whole pool over unsliced.
ENGINE_SLOTS = 461
#: The engine's `max_num_sequences`: the snapshot buffer's row count.
ENGINE_SNAPSHOT_ROWS = 32
#: Row stride of tp1's (and a dep4 rank's) input-projection output `[q|k|v|z|a|b]`.
CKPT_X_STRIDE = 16480

#: A tp4/tep4 rank's GDN conv width (`[q | k | v]` of 4 key and 12 value heads of 128) and
#: the row stride of its input-projection output, `[q | k | v | z | a | b]` = 2560 + 1536 +
#: 12 + 12.
RANK_DIM = 2560
RANK_X_STRIDE = 4120
#: Conv-pool slots of one tep4 rank's stock engine: the larger of the two core-config boots
#: (1784 and 1874; the count is memory-derived), read off the V2 hybrid manager's final
#: "PLE state views bound" line on all four ranks (the PLE and GDN pools share one slot
#: space).
TEP4_SLOTS = 1874
#: GDN layers whose conv states share one slot's page in the V2 manager: a layer's view
#: `[slots, dim, 3]` has slot stride 36 * dim * 3 (276480 at 2560) and layer `i` sits
#: `i * dim * 3` elements into the page.
GDN_PAGE_LAYERS = 36
#: The stock GDN speculative state's snapshot buffer is `[36 layers, max_batch_size,
#: tokens_per_gen_step, dim, 3]`, a layer's view `[max_batch_size, p, dim, 3]`, with
#: `intermediate_state_indices = arange(max_batch_size)` (`GDNReplayState.bind`).
MAX_BATCH_SIZE = 128


def _pinned(batch: int) -> tuple[int, ...]:
    """`batch` distinct slots of the engine pool, slot 0 and the last pinned in."""
    return (0,) + tuple(i * 14 + 3 for i in range(1, batch - 1)) + (ENGINE_SLOTS - 1,)


def _spread(batch: int, pool_slots: int) -> tuple[int, ...]:
    """`batch` distinct slots of a `pool_slots` pool: the top slot (the page's highest
    addresses) and slot 0 first, then a stride-37 walk over the interior (37 is coprime to
    `pool_slots - 2`), so batch row `b` is not slot `b`."""
    rest = tuple((i * 37 + 4) % (pool_slots - 2) + 1 for i in range(max(batch - 2, 0)))
    slots = ((pool_slots - 1, 0) + rest)[:batch]
    assert len(set(slots)) == batch
    return slots


def _tep4(batch: int, seqlen: int) -> dict:
    """A tep4 rank's verify call: 2560 channels of x at row stride 4120, the layer's conv
    view the last of the manager's 36-layer page of 1874 slots, and its snapshots into a
    layer's `[128, seqlen, 2560, 3]` view of the stock buffer."""
    return dict(
        dim=RANK_DIM,
        x_row_stride=RANK_X_STRIDE,
        batch=batch,
        seqlen=seqlen,
        pool_slots=TEP4_SLOTS,
        pool_layers=GDN_PAGE_LAYERS,
        slots=_spread(batch, TEP4_SLOTS),
        snapshots=MAX_BATCH_SIZE,
        snapshot_layers=2,
    )


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

    # spec keys: dim, x_row_stride (default 16480), batch, seqlen, pool_slots,
    # pool_layers (the pool the last layer's view of a page this many layers
    # deep, default 1), slots, snapshots (rows of the snapshot buffer),
    # snapshot_steps (the buffer's step depth, default seqlen), snapshot_layers
    # (the buffer the first layer's view of one this many layers deep, default
    # 1). Every cell is the call site: `x` the token-major q|k|v columns of a
    # [n * p, x_row_stride] projection seen as [n, C, p], bf16, W 4, no bias,
    # SiLU, pad_slot_id -1, no accepted-token roll.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"verify, x over the [n*p, 16480] in_proj, batch 32, {seqlen} tok/row",
            spec=dict(
                dim=CKPT_DIM,
                batch=32,
                seqlen=seqlen,
                pool_slots=ENGINE_SLOTS,
                slots=_pinned(32),
                snapshots=ENGINE_SNAPSHOT_ROWS,
            ),
        )
        # seqlen 8: max_draft_len 7, the widest verify step.
        for seqlen in (3, 8)
    ) + (
        Cell(
            why="seqlen 5 into an 8-step snapshot buffer: steps 5..7 untouched",
            spec=dict(
                dim=CKPT_DIM,
                batch=8,
                seqlen=5,
                pool_slots=ENGINE_SLOTS,
                slots=_pinned(8),
                snapshots=ENGINE_SNAPSHOT_ROWS,
                snapshot_steps=8,
            ),
        ),
        # This target's tep4 member (the MTP unit's): a rank's 4 k / 12 v heads.
        Cell(
            why="tep4 verify, 1 row x 4 tok (MTP3) at 2560 / stride 4120, the 1874-slot page",
            spec=_tep4(1, 4),
        ),
        Cell(
            why="tep4 verify, 128 rows (max_batch_size) x 4 tok (MTP3) at 2560 / stride 4120",
            spec=_tep4(128, 4),
        ),
        Cell(
            why="tep4 verify, 128 rows x 2 tok (draft length 1) at 2560 / stride 4120",
            spec=_tep4(128, 2),
        ),
        Cell(
            why="tep4 verify, 128 rows x 8 tok (draft length 7, the envelope's largest step)",
            spec=_tep4(128, 8),
        ),
    )

    note = """
    Metadata consumed: none.

    Call site: the GDN speculative verify step -- `x` the `[n, 10240, p]`
    transposed view of `[n, p, 10240]` (p = runtime_draft_len + 1, 2..8 here), the layer's
    whole conv pool `[slots, 10240, 3]`, W 4, no bias, `"silu"`, the verify rows'
    slots, `pad_slot_id` -1, snapshots into `[max_num_sequences, p, 10240, 3]`
    with `intermediate_state_indices = arange(n)`, `num_accepted_tokens` None.
    The op advances the pool by all `p` tokens whatever is accepted later; the
    snapshots are what a rollback has to read. Certified at the 461-slot pool
    with a 32-row snapshot buffer and `x` as the target's q|k|v column view
    of the projection: 32 rows at seqlen 3 and at seqlen 8 (the widest verify
    step), and 8 rows at seqlen 5 into an 8-step buffer (a buffer deeper than
    seqlen keeps its later steps untouched); bf16, W 4, no bias, SiLU only.

    This target's tep4 member (the MTP unit's verify, `DecodeTarget._gdn_verify`):
    a rank's 4 k / 12 v heads, so `x` is the `[n, 2560, p]` view of the q|k|v
    columns of the `[n * p, 4120]` in_proj output (strides `(p * 4120, 1,
    4120)`); the pool is one layer's view of the V2 manager's 36-layer conv page
    at the measured 1874 slots (slot stride 276480, the last layer's view); the
    snapshots go to a layer's `[128, p, 2560, 3]` view of the GDN speculative
    state's `[36, max_batch_size, p, 2560, 3]` buffer with
    `intermediate_state_indices = arange(n)`. Certified at 1 row x 4 tokens and
    at 128 rows (max_batch_size) x 2, 4 and 8 tokens (draft lengths 1, 3 and 7;
    128 x 8 = 1024 rows is the envelope's largest verify step). Measured: the
    result has one-ulp bf16 differences in 0 / 2 / 4 / 10 elements of 10240 /
    655360 / 1310720 / 2621440 (relative L2 <= 2.0e-5, cosine >= 0.99999988),
    the pool and the snapshots are bitwise the reference, and no other layer's
    conv page or snapshots, nor x's buffer, is written. GB300 graph replay
    measured 2.0 us a call at 1 x 4 and 4.5 / 5.5 / 8.8 us at 128 x 2 / 4 / 8.

    The kernel is Triton, JIT-compiled at the first call; the cells ran that
    path on sm_103. The x strides, the seqlen and the snapshot strides are
    `tl.constexpr`, so each (p, row stride) the target sends is its own
    compile.

    Not certified: a contiguous `x`, other seqlens, pad-sentinel rows (the
    target never sends one), a dep4 rank (10240 channels at row stride 16480
    over its 850-slot page) and tp4 (no MTP member). The tep4 slot count is the
    larger of two core-config boots (1784 and 1874, memory-derived); no
    MTP-config tep4 engine has been measured. The kernel addresses a slot by
    its index alone, so the pool's size does not change the path.

    Raise loudly, so not guarded: width 1 (triton CompilationError); a missing
    `intermediate_state_indices` with snapshots and slot indices given.

    Not guarded (content): duplicate slots or duplicate snapshot rows race;
    `num_accepted_tokens[b]` must lie in `[1, seqlen]` (the accepted-token
    roll is uncertified; the target passes None). Out-of-range slot values:
    illegal access, not probed.

    Accepted, uncertified: `launch_dependent_kernels` (the target passes
    False). Accepted but not exposed: `cache_seqlens` (ignored by the kernel --
    there is no ring mode), the eagle-tree mask trio, `validate_data`,
    `metadata`, the `_block_n`/`_num_warps`/`_num_stages` overrides. A
    `conv_state` dtype different from `x` is accepted and uncertified.

    Hard limits
      slot offset  int64 in the kernel (`conv_state_indices` loaded `.to(tl.int64)`):
                   at the 1874-slot page the last slot's offset is 517.8 M
                   elements, below 2^31 either way.
      x offset     `row * stride_x_seq` in int32: 128 x 8 x 4120 = 4.2 M
                   elements at the envelope, far below 2^31.
      none probed beyond the cells.
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
