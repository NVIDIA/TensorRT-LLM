# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A verify step's multi-token causal conv1d written token-major (`[n * p, C]`, the layout the GDN
cached replay reads): bitwise the family `causal_conv1d_triton_update` plus a token-major copy.
With `wait_producer` the kernel reads the slots and the conv pool before `griddepcontrol.wait`
and `x` after it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.ssm.causal_conv1d_triton_update import (  # noqa: E501
    causal_conv1d_triton_update,
)
from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber
from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import _causal_conv1d_update_kernel
from tensorrt_llm._utils import get_sm_version

_BLOCK_N = 128
_NUM_WARPS = 4
#: Sequences a `_verify_conv_seqs_kernel` program (taps loaded once): 0 takes `_AUTO_SEQS` from
#: `_AUTO_MIN_SEQUENCES` sequences on and the family kernel below; a positive value forces it.
_SEQS = 0
_AUTO_SEQS = 2
_AUTO_MIN_SEQUENCES = 104


def _seqs_a_program(batch: int) -> int:
    if _SEQS > 0:
        return _SEQS
    return _AUTO_SEQS if batch >= _AUTO_MIN_SEQUENCES else 1


@triton.jit(do_not_specialize=["batch"])
def _verify_conv_seqs_kernel(
    x_ptr,
    w_ptr,
    conv_state_ptr,
    slots_ptr,
    window_ptr,
    window_rows_ptr,
    o_ptr,
    batch,
    dim: tl.constexpr,
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_cs_seq: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    stride_slots: tl.constexpr,
    stride_win_seq: tl.constexpr,
    stride_win_step: tl.constexpr,
    stride_win_dim: tl.constexpr,
    stride_win_win: tl.constexpr,
    stride_wrows: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    pad_slot_id: tl.constexpr,
    SEQLEN: tl.constexpr,
    SEQS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LAUNCH_DEPENDENT_KERNELS: tl.constexpr,
    WAIT_PRODUCER: tl.constexpr,
):
    """`_causal_conv1d_update_kernel` (width 4, SiLU, pad slots, snapshots, `SEQLEN >= 3`) over
    `SEQS` sequences a program, per element the same arithmetic and stores."""
    if LAUNCH_DEPENDENT_KERNELS:
        tl.extra.cuda.gdc_launch_dependents()
    seq0 = tl.program_id(0) * SEQS
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim
    w_base = w_ptr + (idx_feats * stride_w_dim)
    w_col0 = tl.load(w_base + (0 * stride_w_width), mask_w, other=0.0)
    w_col1 = tl.load(w_base + (1 * stride_w_width), mask_w, other=0.0)
    w_col2 = tl.load(w_base + (2 * stride_w_width), mask_w, other=0.0)
    w_col3 = tl.load(w_base + (3 * stride_w_width), mask_w, other=0.0)
    # Slots, snapshot rows and pre-step windows (earlier steps' data) before the dependency wait.
    slots = ()
    wrows = ()
    lives = ()
    cols = ()
    for s in tl.static_range(SEQS):
        seq = seq0 + s
        in_batch = seq < batch
        slot = tl.load(slots_ptr + seq * stride_slots, mask=in_batch, other=pad_slot_id).to(
            tl.int64
        )
        wrow = tl.load(window_rows_ptr + seq * stride_wrows, mask=in_batch, other=0).to(tl.int64)
        live = in_batch & (slot != pad_slot_id)
        m = mask_w & live
        prior = conv_state_ptr + (slot * stride_cs_seq) + (idx_feats * stride_cs_dim)
        cols = cols + (
            tl.load(prior, m, 0.0),
            tl.load(prior + 1 * stride_cs_tok, m, 0.0),
            tl.load(prior + 2 * stride_cs_tok, m, 0.0),
        )
        slots = slots + (slot,)
        wrows = wrows + (wrow,)
        lives = lives + (live,)
    if WAIT_PRODUCER:
        gdc_wait_with_memory_clobber()
    xs = ()
    for s in tl.static_range(SEQS):
        x_base = x_ptr + ((seq0 + s) * stride_x_seq) + (idx_feats * stride_x_dim)
        for t in tl.static_range(SEQLEN):
            xs = xs + (tl.load(x_base + t * stride_x_token, mask=mask_w & lives[s]),)
    tl.debug_barrier()
    for s in tl.static_range(SEQS):
        seq = seq0 + s
        m = mask_w & lives[s]
        # The advanced pool: the last three inputs.
        state_base = conv_state_ptr + (slots[s] * stride_cs_seq) + (idx_feats * stride_cs_dim)
        for k in tl.static_range(3):
            tl.store(state_base + k * stride_cs_tok, xs[s * SEQLEN + SEQLEN - 3 + k], m)
        col0 = cols[3 * s]
        col1 = cols[3 * s + 1]
        col2 = cols[3 * s + 2]
        for t in tl.static_range(SEQLEN):
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            acc += col0 * w_col0
            acc += col1 * w_col1
            acc += col2 * w_col2
            x_t = xs[s * SEQLEN + t]
            acc += x_t * w_col3
            col0 = col1
            col1 = col2
            col2 = x_t
            win = (
                window_ptr
                + wrows[s] * stride_win_seq
                + t * stride_win_step
                + idx_feats * stride_win_dim
            )
            tl.store(win + 0 * stride_win_win, col0, mask=m)
            tl.store(win + 1 * stride_win_win, col1, mask=m)
            tl.store(win + 2 * stride_win_win, col2, mask=m)
            acc = acc / (1 + tl.exp(-acc))
            tl.store(
                o_ptr + seq * stride_o_seq + t * stride_o_token + idx_feats * stride_o_dim,
                acc,
                mask=m,
            )


def _launch_seqs(
    x3, conv_state, weight, slots, window, window_rows, pad_slot_id, out3, ldk, wait, seqs
):
    batch, dim, seqlen = x3.shape
    _verify_conv_seqs_kernel[(triton.cdiv(batch, seqs), triton.cdiv(dim, _BLOCK_N))](
        x3,
        weight,
        conv_state,
        slots,
        window,
        window_rows,
        out3,
        batch,
        dim,
        *x3.stride(),
        *weight.stride(),
        *conv_state.stride(),
        slots.stride(0),
        *window.stride(),
        window_rows.stride(0),
        *out3.stride(),
        pad_slot_id,
        SEQLEN=seqlen,
        SEQS=seqs,
        BLOCK_N=_BLOCK_N,
        LAUNCH_DEPENDENT_KERNELS=ldk,
        WAIT_PRODUCER=wait,
        num_warps=_NUM_WARPS,
        **({"launch_pdl": True} if wait else {}),
    )


def _launch(x3, conv_state, weight, slots, window, window_rows, pad_slot_id, out3, ldk, wait):
    batch, dim, seqlen = x3.shape
    width = weight.shape[1]
    seqs = _seqs_a_program(batch)
    if seqs > 1 and width == 4 and seqlen >= 3 and conv_state.shape[-1] == 3:
        return _launch_seqs(
            x3, conv_state, weight, slots, window, window_rows, pad_slot_id, out3, ldk, wait, seqs
        )
    state_len = width - 1
    # `HOIST_X` (width 4) loads every token's x row before the stores, which ptxas cannot prove
    # do not alias them: one L2 round trip for x a row instead of p. Bitwise.
    hoist = width == 4
    _causal_conv1d_update_kernel[(batch, triton.cdiv(dim, _BLOCK_N))](
        x3,
        weight,
        None,
        conv_state,
        None,
        slots,
        None,
        window,
        window_rows,
        None,
        None,
        None,
        out3,
        batch,
        dim,
        seqlen,
        state_len,
        conv_state.shape[0],
        *x3.stride(),
        *weight.stride(),
        *conv_state.stride(),
        slots.stride(0),
        *window.stride(),
        window_rows.stride(0),
        0,
        0,
        0,
        0,
        0,
        0,
        *out3.stride(),
        pad_slot_id,
        HAS_BIAS=False,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=True,
        IS_CONTINUOUS_BATCHING=True,
        IS_SPEC_DECODING=False,
        NP2_STATELEN=triton.next_power_of_2(state_len),
        NP2_SEQLEN=triton.next_power_of_2(seqlen),
        USE_PAD_SLOT=True,
        BLOCK_N=_BLOCK_N,
        SAVE_INTERMEDIATE=True,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=False,
        LAUNCH_DEPENDENT_KERNELS=ldk,
        num_warps=_NUM_WARPS,
        **({"WAIT_PRODUCER": True, "launch_pdl": True} if wait else {}),
        **({"HOIST_X": True} if hoist else {}),
    )


class CausalConv1dTokenMajor(OpWrapper):
    """`out = causal_conv1d_token_major(x, rows, conv_state, weight, slots, window,
    window_rows, pad_slot_id, launch_dependent_kernels, wait_producer)`.

    `x` is the `[rows * p, C]` q|k|v column view of the projection; the step is the family's
    `causal_conv1d_triton_update` over its `[rows, C, p]` view (SiLU, no bias,
    `conv_state_indices=slots`, snapshots into `window` at `window_rows`). Returns the
    contiguous token-major `[rows * p, C]` output.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="tep4 verify, 1 request x 3 tokens (MTP-2 c=1)", spec=dict(rows=1, p=3)),
        Cell(why="tep4 verify, 4 requests x 4 tokens (MTP-3)", spec=dict(rows=4, p=4)),
        Cell(why="tep4 verify, 1 request x 3 tokens, PDL wait", spec=dict(rows=1, p=3, wait=True)),
    )

    note = """
    Source: the Triton `_causal_conv1d_update_kernel` (`modules/mamba/causal_conv1d_triton.py`)
    with token-major output strides, and `_verify_conv_seqs_kernel` above. Certified bitwise
    against the family entry plus the copy, pool and snapshots included.
    """

    def __call__(
        self,
        x: torch.Tensor,
        rows: int,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        slots: torch.Tensor,
        window: torch.Tensor,
        window_rows: torch.Tensor,
        pad_slot_id: int,
        launch_dependent_kernels: bool = False,
        wait_producer: bool = False,
    ) -> torch.Tensor:
        tokens, dim = x.shape
        p = tokens // rows
        out = torch.empty([tokens, dim], dtype=x.dtype, device=x.device)
        pdl = get_sm_version() >= 90
        _launch(
            torch.transpose(torch.reshape(x, [rows, p, dim]), 1, 2),
            conv_state,
            weight,
            slots,
            window,
            window_rows,
            pad_slot_id,
            torch.transpose(torch.reshape(out, [rows, p, dim]), 1, 2),
            launch_dependent_kernels and pdl,
            wait_producer and pdl,
        )
        return out

    def reference(
        self,
        x: torch.Tensor,
        rows: int,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        slots: torch.Tensor,
        window: torch.Tensor,
        window_rows: torch.Tensor,
        pad_slot_id: int,
        launch_dependent_kernels: bool = False,
        wait_producer: bool = False,
    ) -> torch.Tensor:
        """The family entry over `x`'s `[rows, C, p]` view, copied token-major (it advances
        `conv_state` and writes `window` as the op does: call it on copies)."""
        tokens, dim = x.shape
        p = tokens // rows
        out = causal_conv1d_triton_update(
            x=torch.transpose(torch.reshape(x, [rows, p, dim]), 1, 2),
            conv_state=conv_state,
            weight=weight,
            bias=None,
            activation="silu",
            conv_state_indices=slots,
            pad_slot_id=pad_slot_id,
            num_accepted_tokens=None,
            intermediate_conv_window=window,
            intermediate_state_indices=window_rows,
        )
        return torch.reshape(torch.transpose(out, 1, 2), [tokens, dim])

    def is_valid(self, x, rows, conv_state, weight, slots, window, window_rows, *args, **kwargs):
        assert x.dim() == 2 and x.stride(1) == 1 and x.shape[0] % rows == 0, (
            "x is the [rows * p, C] column view of the projection, unit column stride"
        )
        # The pool is read through its strides; the taps must be unit-stride along the width.
        assert weight.stride(1) == 1, "conv taps [C, W] with a unit width stride"


causal_conv1d_token_major = CausalConv1dTokenMajor()
