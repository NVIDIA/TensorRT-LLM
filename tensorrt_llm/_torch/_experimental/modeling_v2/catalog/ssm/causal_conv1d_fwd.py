# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal depthwise conv1d over a whole (ragged) batch of sequences, with each
sequence's convolution state read from and published to a paged state pool in
the same call."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: qwen4_exp's GDN mixer: `[q|k|v]` is 10240 wide, the conv is width 4, so each
#: pool slot holds a 3-column window.
QWEN4_EXP_DIM = 10240
#: The largest conv-pool slot count the cache manager was seen to allocate for
#: qwen4_exp (10, 35, 366, 368, 396, 399, 432 and 461 across twelve
#: runtime-value dumps, while `max_num_sequences` never exceeds 32). The target
#: hands the pool over unsliced, so 461 tops the certified interval.
QWEN4_EXP_SLOTS = 461

#: Nemotron-3 Ultra's Mamba2 mixer at tep4, per rank: the conv runs over
#: xBC = x 4096 | B 256 | C 256 channels, width 4, with bias.
ULTRA_TP4_DIM = 4608
#: It reads xBC in place from the rank's in_proj output row, laid out
#: z 4096 | xBC 4608 | dt 64: a column slice with token stride 8768.
ULTRA_TP4_PROJ_WIDTH = 8768
ULTRA_TP4_XBC_OFFSET = 4096
#: The conv pool MambaHybridCacheManagerV2 allocates at that geometry
#: (max_batch_size 32), measured: 35 slots, each slot holding all 48 Mamba
#: layers' `[4608, 3]` windows back to back, so one layer's view is
#: `[35, 4608, 3]` with strides `(48 * 4608 * 3, 3, 1)`.
ULTRA_TP4_SLOTS = 35
ULTRA_TP4_POOL_LAYERS = 48

#: At dep4 (attention DP) every rank runs the whole mixer: xBC = x 16384 |
#: B 1024 | C 1024 channels, read from the in_proj row z 16384 | xBC 18432 |
#: dt 256 (token stride 35072). The cache manager then uses tp=1 for Mamba;
#: measured at that geometry: the same 35 slots x 48 layers, one layer's view
#: `[35, 18432, 3]` with strides `(48 * 18432 * 3, 3, 1)`.
ULTRA_DEP4_DIM = 18432
ULTRA_DEP4_PROJ_WIDTH = 35072
ULTRA_DEP4_XBC_OFFSET = 16384

#: At tep8 (the BF16 checkpoint over 8 ranks) a rank runs an eighth of the mixer: xBC =
#: x 2048 | B 128 | C 128 channels of the in_proj row z 2048 | xBC 2304 | dt 32 (token
#: stride 4384), one layer's pool view `[35, 2304, 3]` with strides `(48 * 2304 * 3, 3, 1)`.
ULTRA_TEP8_DIM = 2304
ULTRA_TEP8_PROJ_WIDTH = 4384
ULTRA_TEP8_XBC_OFFSET = 2048
#: The engine's pool at max_batch_size 256, the tep8 target's serving envelope: 271 slots.
ULTRA_TEP8_SLOTS_256 = 271


def _engine_batch_slots() -> tuple[int, ...]:
    """32 distinct slots over qwen4_exp's whole engine pool, slot 0 and the last included."""
    slots = (0, QWEN4_EXP_SLOTS - 1) + tuple(
        (i * 13 + 5) % (QWEN4_EXP_SLOTS - 2) + 1 for i in range(30)
    )
    assert len(set(slots)) == 32
    return slots


def _ultra256_batch_slots() -> tuple[int, ...]:
    """256 distinct, scattered slots of the 271-slot pool, slot 0 and the last included."""
    slots = (0, ULTRA_TEP8_SLOTS_256 - 1) + tuple(
        (i * 7 + 3) % (ULTRA_TEP8_SLOTS_256 - 2) + 1 for i in range(254)
    )
    assert len(set(slots)) == 256
    return slots


def _ultra_batch_slots() -> tuple[int, ...]:
    """32 distinct, scattered slots of the 35-slot Ultra pool, slot 0 and the last included."""
    slots = (0, ULTRA_TP4_SLOTS - 1) + tuple(
        (i * 7 + 3) % (ULTRA_TP4_SLOTS - 2) + 1 for i in range(30)
    )
    assert len(set(slots)) == 32
    return slots


#: The Mamba2 prefill call site: xBC read in place from the projection, the
#: result written token-major, bias on, one layer's view of the engine pool.
_ULTRA = dict(
    dim=ULTRA_TP4_DIM,
    bias=True,
    layout="proj",
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


class CausalConv1dFwd(OpWrapper):
    """Depthwise causal conv1d (+ optional SiLU) of every sequence in a batch,
    against its carried window, publishing the new window -- one launch.

    For sequence `s` over tokens `[b, e)` and `slot = cache_indices[s]` (or `s`):
    `xpad = cat([conv_states[slot] if has_initial_state[s] else 0, x[:, b:e]])`,
    `out[c, b + t] = act(sum_k weight[c, k] * xpad[c, t + k] + bias[c])` with the
    taps accumulated in fp32 and rounded once, and
    `conv_states[slot] = xpad[:, L : L + W - 1]` -- the raw input tail, not the
    activation, exact in every dtype.

    Fusion boundary: the state read, the convolution, the bias, the SiLU and the
    state publish are inside. The projection that produces `x`, and slicing the
    result into q/k/v or x/B/C, are the caller's.

    Writes the activation into `out`, or into `x` in place when `out is None`,
    and returns that tensor. `conv_states` is advanced in place for every
    non-padded sequence and is not returned.

    Two kernels sit behind the op, selected from `x`'s layout: unit stride on
    the token axis is channel-major (in place allowed; `out`, if given, must
    have `x`'s exact strides); unit stride on the channel axis with a token
    stride > 1, `dim > 1` and at least one token is channel-last (`out`
    required, channel-last, disjoint from `x`). The two agree bitwise.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103, Arch.SM_107})

    # spec keys: dim, width, dtype, lengths (varlen) or batch3d=(batch, seqlen),
    # pool_slots (None -> no conv_states), pool_layers/pool_layer (the pool is
    # layer `pool_layer`'s view of a `[slots, pool_layers, dim, W-1]` buffer),
    # slots (None -> cache_indices None), initial (None -> has_initial_state
    # None), silu, bias, pad_slot_id, layout: "in_place" | "out" |
    # "channel_last" | "proj" (x the `.t()` of columns [proj_offset,
    # proj_offset + dim) of a token-major `[T, proj_width]` buffer, out a
    # token-major `[T, dim]` buffer's `.t()`).
    CELLS: tuple[Cell, ...] = (
        (
            Cell(
                why="qwen4_exp GDN prefill 37+1+128 tokens in place, 8-slot pool",
                spec=dict(
                    dim=QWEN4_EXP_DIM,
                    lengths=(37, 1, 128),
                    pool_slots=8,
                    slots=(3, 0, 5),
                    initial=(True, False, True),
                ),
            ),
            Cell(
                why="qwen4_exp GDN prefill at its 461-slot pool: slots 460, 0, 230",
                spec=dict(
                    dim=QWEN4_EXP_DIM,
                    lengths=(37, 1, 128),
                    pool_slots=QWEN4_EXP_SLOTS,
                    slots=(QWEN4_EXP_SLOTS - 1, 0, QWEN4_EXP_SLOTS // 2),
                    initial=(True, False, True),
                ),
            ),
            Cell(
                why="32 sequences (the batch ceiling) scattered over the 461-slot pool",
                spec=dict(
                    dim=QWEN4_EXP_DIM,
                    lengths=tuple(1 + (i * 7) % 96 for i in range(32)),
                    pool_slots=QWEN4_EXP_SLOTS,
                    slots=_engine_batch_slots(),
                    initial=tuple(i % 3 != 0 for i in range(32)),
                ),
            ),
            Cell(
                why="explicit channel-major out: result lands there, x is read-only",
                spec=dict(
                    dim=QWEN4_EXP_DIM,
                    lengths=(64, 9),
                    pool_slots=4,
                    slots=(1, 2),
                    initial=(True, True),
                    layout="out",
                ),
            ),
            Cell(
                why="channel-last x (a bare .t() of token-major), into its own out",
                spec=dict(
                    dim=QWEN4_EXP_DIM,
                    lengths=(37, 1, 128),
                    pool_slots=8,
                    slots=(3, 0, 5),
                    initial=(True, False, True),
                    layout="channel_last",
                ),
            ),
            Cell(
                why="Mamba2 tep4 prefill: 4 sequences, mixed carry, scattered engine slots",
                spec=dict(
                    _ULTRA,
                    lengths=(37, 1, 128, 300),
                    slots=(ULTRA_TP4_SLOTS - 1, 0, 17, 5),
                    initial=(True, False, True, False),
                ),
            ),
            Cell(
                why="Mamba2 tep4: 32 sequences (batch ceiling) over the 35-slot pool",
                spec=dict(
                    _ULTRA,
                    lengths=tuple(1 + (i * 37) % 255 for i in range(32)),
                    slots=_ultra_batch_slots(),
                    initial=tuple(i % 3 != 0 for i in range(32)),
                ),
            ),
            Cell(
                why="Mamba2 tep4: one 4101-token sequence across many token chunks",
                spec=dict(_ULTRA, lengths=(4101,), slots=(12,), initial=(False,)),
            ),
            Cell(
                why="Mamba2 tep4 at the 8192-token step ceiling, 3 sequences",
                spec=dict(
                    _ULTRA, lengths=(4096, 4095, 1), slots=(3, 30, 0), initial=(True, False, True)
                ),
            ),
            Cell(
                why="Mamba2 tep4 chunked-prefill continuations shorter than the window",
                spec=dict(_ULTRA, lengths=(1, 2, 3, 5), slots=(1, 2, 3, 4), initial=(True,) * 4),
            ),
            Cell(
                why="Mamba2 tep4 layout, pad_slot_id -1: the channel-last kernel skips it",
                spec=dict(_ULTRA, lengths=(16, 8, 16), slots=(0, -1, 2), initial=(True,) * 3),
            ),
            Cell(
                why="Mamba2 dep4 prefill: 4 sequences, mixed carry, scattered engine slots",
                spec=dict(
                    _ULTRA_DEP4,
                    lengths=(37, 1, 128, 300),
                    slots=(ULTRA_TP4_SLOTS - 1, 0, 17, 5),
                    initial=(True, False, True, False),
                ),
            ),
            Cell(
                why="Mamba2 dep4: 32 sequences (batch ceiling) over the 35-slot pool",
                spec=dict(
                    _ULTRA_DEP4,
                    lengths=tuple(1 + (i * 37) % 255 for i in range(32)),
                    slots=_ultra_batch_slots(),
                    initial=tuple(i % 3 != 0 for i in range(32)),
                ),
            ),
            Cell(
                why="Mamba2 dep4 at the 8192-token step ceiling, 3 sequences",
                spec=dict(
                    _ULTRA_DEP4,
                    lengths=(4096, 4095, 1),
                    slots=(3, 30, 0),
                    initial=(True, False, True),
                ),
            ),
            Cell(
                why="Mamba2 dep4 chunked-prefill continuations shorter than the window",
                spec=dict(
                    _ULTRA_DEP4, lengths=(1, 2, 3, 5), slots=(1, 2, 3, 4), initial=(True,) * 4
                ),
            ),
            Cell(
                why="Mamba2 tep8 prefill: 4 sequences, mixed carry, scattered engine slots",
                spec=dict(
                    _ULTRA_TEP8,
                    lengths=(37, 1, 128, 300),
                    slots=(ULTRA_TP4_SLOTS - 1, 0, 17, 5),
                    initial=(True, False, True, False),
                ),
            ),
            Cell(
                why="Mamba2 tep8: 32 sequences (batch ceiling) over the 35-slot pool",
                spec=dict(
                    _ULTRA_TEP8,
                    lengths=tuple(1 + (i * 37) % 255 for i in range(32)),
                    slots=_ultra_batch_slots(),
                    initial=tuple(i % 3 != 0 for i in range(32)),
                ),
            ),
            Cell(
                why="Mamba2 tep8: 256 sequences (the serving envelope's batch ceiling) of up to "
                "32 tokens, one 8192-token step, over the 271-slot pool",
                spec=dict(
                    _ULTRA_TEP8,
                    lengths=tuple(1 + (i * 13) % 32 for i in range(256)),
                    pool_slots=ULTRA_TEP8_SLOTS_256,
                    slots=_ultra256_batch_slots(),
                    initial=tuple(i % 3 != 0 for i in range(256)),
                ),
            ),
            Cell(
                why="Mamba2 tep8 at the 8192-token step ceiling, 3 sequences",
                spec=dict(
                    _ULTRA_TEP8,
                    lengths=(4096, 4095, 1),
                    slots=(3, 30, 0),
                    initial=(True, False, True),
                ),
            ),
            Cell(
                why="Mamba2 tep8 chunked-prefill continuations shorter than the window",
                spec=dict(
                    _ULTRA_TEP8, lengths=(1, 2, 3, 5), slots=(1, 2, 3, 4), initial=(True,) * 4
                ),
            ),
            Cell(
                why="Mamba2 tep8 layout, pad_slot_id -1: the channel-last kernel skips it",
                spec=dict(_ULTRA_TEP8, lengths=(16, 8, 16), slots=(0, -1, 2), initial=(True,) * 3),
            ),
            Cell(
                why="pad_slot_id -1 (the engine's sentinel): sequence and slot skipped",
                spec=dict(
                    dim=1024,
                    lengths=(16, 8, 16),
                    pool_slots=4,
                    slots=(0, -1, 2),
                    initial=(True, True, True),
                ),
            ),
            Cell(
                why="pad_slot_id 99: the sentinel is compared by value",
                spec=dict(
                    dim=1024,
                    lengths=(16, 8, 16),
                    pool_slots=4,
                    slots=(0, 99, 2),
                    initial=(True, True, True),
                    pad_slot_id=99,
                ),
            ),
            Cell(
                why="sequences shorter than the window: state part carry, part tokens",
                spec=dict(
                    dim=512,
                    lengths=(2, 1, 2),
                    pool_slots=4,
                    slots=(0, 1, 2),
                    initial=(True, True, False),
                ),
            ),
            Cell(
                why="a zero-token sequence republishes its carry unchanged",
                spec=dict(
                    dim=512, lengths=(0, 40), pool_slots=4, slots=(2, 3), initial=(True, False)
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"{str(dtype).removeprefix('torch.')} with bias, silu={silu}",
                spec=dict(
                    dim=256,
                    lengths=(30, 34),
                    pool_slots=2,
                    slots=(0, 1),
                    initial=(True, True),
                    dtype=dtype,
                    bias=True,
                    silu=silu,
                ),
            )
            for dtype in (torch.bfloat16, torch.float16, torch.float32)
            for silu in (True, False)
        )
        + (
            Cell(
                why="conv_states None: every sequence starts from zeros",
                spec=dict(dim=512, lengths=(12, 20), pool_slots=None, slots=None, initial=None),
            ),
            Cell(
                why="cache_indices/has_initial_state None: slot s, no carry",
                spec=dict(dim=512, lengths=(24, 40), pool_slots=2, slots=None, initial=None),
            ),
            Cell(
                why="dense [batch, dim, seqlen] layout, query_start_loc None",
                spec=dict(
                    dim=256, batch3d=(3, 40), pool_slots=3, slots=None, initial=(True, False, True)
                ),
            ),
            Cell(
                why="one 2048-token sequence: the vectorised channel-major load",
                spec=dict(
                    dim=QWEN4_EXP_DIM, lengths=(2048,), pool_slots=1, slots=(0,), initial=(True,)
                ),
            ),
            Cell(
                why="one 2047-token sequence: the scalar channel-major load",
                spec=dict(
                    dim=QWEN4_EXP_DIM, lengths=(2047,), pool_slots=1, slots=(0,), initial=(True,)
                ),
            ),
            Cell(
                why="several long sequences: the 128-thread block",
                spec=dict(
                    dim=1024,
                    lengths=(700, 701, 648),
                    pool_slots=3,
                    slots=(0, 1, 2),
                    initial=(True, True, True),
                ),
            ),
            Cell(
                why="every sequence a single token",
                spec=dict(
                    dim=1024,
                    lengths=(1, 1, 1, 1),
                    pool_slots=4,
                    slots=(0, 1, 2, 3),
                    initial=(True, True, True, True),
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"conv width {width}",
                spec=dict(
                    dim=512,
                    width=width,
                    lengths=(33, 7),
                    pool_slots=2,
                    slots=(0, 1),
                    initial=(True, True),
                ),
            )
            for width in (2, 3)
        )
    )

    note = """
    Metadata consumed: none. Batch layout (`query_start_loc`, int32), pool
    (`conv_states`, `[slots, dim, W - 1]`), slot map (`cache_indices`, int32)
    and carry flags (`has_initial_state`, bool) are explicit arguments.

    Call sites.
    qwen4_exp, the GDN prefill half: `x` the `[10240, num_prefill_tokens]`
    channel-major output of `extract_transpose_prefill_slice`, W 4, no bias,
    SiLU on, the layer's whole conv pool unsliced, `pad_slot_id` -1, `out`
    None. The `qwen4_exp GDN` cells and the 461-slot batch cell are that call;
    461 slots tops the certified pool interval (8 and 461 bracket every width
    the cache manager was seen to allocate). At 461 the activation and state
    are bitwise identical to the 8-slot result for the same slot content.
    nemotron_h Ultra tep4, the Mamba2 prefill: `x = proj[:T, 4096:8704].t()`
    -- the rank's xBC columns of the in_proj output read in place (channel-last,
    token stride 8768) -- into `out = torch.empty(T, 4608).t()`, so the result
    is token-major and x/B/C are its column views; W 4 with bias, SiLU on;
    `conv_states` the layer's engine view `[35, 4608, 3]` with slot stride
    48 * 4608 * 3 (the 48 Mamba layers' windows share one slot);
    `query_start_loc`, `cache_indices` and `has_initial_state` the context
    prefix of the step's Mamba2 metadata buffers; `pad_slot_id` -1. The
    `Mamba2 tep4` cells are that call; the in_proj buffer outside the read and
    the other 47 layers' windows are gated bitwise unchanged.
    nemotron_h Ultra dep4, the same call unsharded: `x = proj[:T,
    16384:34816].t()` (token stride 35072), `out = torch.empty(T, 18432).t()`,
    `conv_states` the layer's view `[35, 18432, 3]`, slot stride
    48 * 18432 * 3 -- the `Mamba2 dep4` cells.

    Preconditions the op checks itself (all raise): dtypes in {bf16, fp16,
    fp32}, `conv_states`/`bias` dtype, int32 `query_start_loc`/`cache_indices`,
    bool `has_initial_state` of length batch, `out` dtype/shape/layout for the
    selected kernel, channel-last in place, `x` rank.

    Not checked, not guarded (content, not metadata, or an illegal access
    rather than a wrong number): a `cache_indices` value outside `[0, slots)`;
    `has_initial_state[s]` True with `conv_states=None` (reads through null);
    `conv_states`' shape (only its dtype is validated). Sequences of length 0
    and shorter than `W - 1` are legal. Unnamed slots are bitwise untouched.

    `TRTLLM_CAUSAL_CONV1D_SILU_MODE` selects faster SiLU forms on the
    channel-last kernel; only the default (unset) is certified.

    Hard limits
      x token/channel strides  uint32 stride fields  boundary 2^32 elements
        not probed.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        conv_states: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        silu_activation: bool,
        pad_slot_id: int,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch.ops.trtllm.causal_conv1d_fwd(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            silu_activation,
            pad_slot_id,
            out,
        )
        return x if out is None else out

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        conv_states: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        silu_activation: bool,
        pad_slot_id: int,
        out: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """`(activation, conv_states)` as the call leaves them; mutates nothing.

        The activation starts from `out` (or `x`) so a padded sequence's
        columns keep whatever was there, as the op leaves them.
        """
        width = weight.shape[-1]
        act = (x if out is None else out).clone()
        states = None if conv_states is None else conv_states.clone()
        if query_start_loc is not None:
            starts = query_start_loc.tolist()
            spans = [(slice(None), slice(starts[s], starts[s + 1])) for s in range(len(starts) - 1)]
        else:
            spans = [(s,) for s in range(x.shape[0])]
        slots = None if cache_indices is None else cache_indices.tolist()
        carry = None if has_initial_state is None else has_initial_state.tolist()
        for s, where in enumerate(spans):
            slot = s if slots is None else slots[s]
            if slots is not None and slot == pad_slot_id:
                continue
            tokens = x[where].float()  # [dim, L]
            length = tokens.shape[1]
            if conv_states is not None and carry is not None and carry[s]:
                prefix = conv_states[slot].float()
            else:
                prefix = tokens.new_zeros(tokens.shape[0], width - 1)
            padded = torch.cat([prefix, tokens], dim=1)
            acc = torch.zeros_like(tokens)
            for k in range(width):
                acc = acc + weight[:, k].float().unsqueeze(1) * padded[:, k : k + length]
            if bias is not None:
                acc = acc + bias.float().unsqueeze(1)
            if silu_activation:
                acc = acc * torch.sigmoid(acc)
            if length:
                act[where] = acc.to(act.dtype)
            if states is not None:
                states[slot] = padded[:, length : length + width - 1].to(states.dtype)
        return act, states

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        conv_states: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        silu_activation: bool,
        pad_slot_id: int,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        # Widths outside 2..4 fall off the dispatcher with no launch at all:
        # measured at W 1 and W 5, destination and pool bitwise unchanged.
        assert weight.shape[-1] in (2, 3, 4), (
            f"width {weight.shape[-1]} has no kernel; only 2, 3 and 4 are dispatched, "
            "and any other width is a silent no-op"
        )
        # The dispatch reads weight (and bias) as x's type without converting:
        # an fp32 weight with bf16 x moved the result by 11.5.
        assert weight.dtype == x.dtype, (
            "weight must have x's dtype; the kernel reads it as x's type without converting"
        )
        # Both kernels read entry `b` of the three per-sequence tensors at
        # data_ptr + b, ignoring their strides: a stride-2 view of each was
        # measured 2.9 (cache_indices), 3.2 (has_initial_state) and 4.6
        # (query_start_loc) off, with no error.
        for name, meta in (
            ("query_start_loc", query_start_loc),
            ("cache_indices", cache_indices),
            ("has_initial_state", has_initial_state),
        ):
            assert meta is None or meta.numel() < 2 or meta.stride(-1) == 1, (
                f"{name} must have unit stride; the kernels read entry b at data_ptr + b "
                "and ignore the stride, so a strided view silently selects the wrong entries"
            )
        varlen = query_start_loc is not None
        c_axis, l_axis = (0, 1) if varlen else (1, 2)
        if x.dim() != (2 if varlen else 3):
            return  # the op rejects a wrong rank itself
        channel_last = (
            x.stride(c_axis) == 1
            and x.stride(l_axis) > 1
            and x.shape[c_axis] > 1
            and x.shape[l_axis] > 0
        )
        if not channel_last:
            # The channel-major kernel walks the token axis and the state's last
            # axis contiguously, ignoring both strides: token stride 2 landed
            # 16.3 off; a state last-dim stride 2 landed 7.9 off (5.4 in the
            # published state). The channel-last kernel honours every stride.
            assert x.shape[l_axis] == 0 or x.stride(l_axis) == 1, (
                "the channel-major kernel loads the token axis contiguously; "
                "x must have stride 1 along it"
            )
            assert conv_states is None or conv_states.stride(-1) == 1, (
                "the channel-major kernel ignores conv_states' last-dim stride; it must be 1"
            )


causal_conv1d_fwd = CausalConv1dFwd()
