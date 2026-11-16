# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection post-block injection plus the next site's per-(row, stream) RMS
statistic, without the normalized copy: `qwen4_exp_hc_inject_norm` with its
`normed` store replaced by one fp32 rstd per (row, stream).

Tier 5 (Triton written for this target). At prefill sizes the grouped norm's output
(`[T, C*H]`, 335 MB at 16384 rows) is a full extra write whose only readers are the
site's two projections; those form `bf16(hc_out * rstd * (1 + w))` themselves
(`qwen4_exp_hc_first_rescale_tc` on its A tiles, `qwen4_exp_hc_up_gate_mean_rstd` in
its epilogue), so the residual pass moves the residual in and out once and nothing
else.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Rows from which a wide (`hidden >= 2048`) launch runs two warps a program instead
#: of eight (`qwen4_exp_hc_inject_norm`'s split; the reduction tree follows it).
_FEW_WARPS_MIN_ROWS = 512


@triton.jit
def _hc_inject_rstd_kernel(
    hc_ptr,
    block_ptr,
    inject_ptr,
    shared_ptr,
    gate_ptr,
    hc_out_ptr,
    rstd_ptr,
    stride_hc,
    stride_block,
    stride_inject,
    stride_shared,
    stride_gate,
    stride_out,
    eps,
    HC_COUNT: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_INJECT: tl.constexpr,
    HAS_SHARED: tl.constexpr,
):
    # One program per (row, stream), stream-minor: a row's block output (and the
    # combine's operands) come from HBM once and from L2 for the other programs. The
    # loads, the fp32 arithmetic and the reduction are `qwen4_exp_hc_inject_norm`'s
    # (without the fold), so `hc_out` and rstd are bitwise what that kernel stores and
    # multiplies by at the same warp count.
    pid = tl.program_id(0).to(tl.int64)
    row = pid // HC_COUNT
    stream = (pid % HC_COUNT).to(tl.int32)
    cols = tl.arange(0, BLOCK)
    mask = cols < HIDDEN
    offs = stream * HIDDEN + cols
    x = tl.load(hc_ptr + row * stride_hc + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_INJECT:
        b = tl.load(block_ptr + row * stride_block + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_SHARED:
            # The MoE block output, `routed + sigmoid(gate) * shared` in fp32, rounded
            # once to the routed output's dtype.
            sg = tl.sigmoid(tl.load(gate_ptr + row * stride_gate).to(tl.float32))
            sh = tl.load(shared_ptr + row * stride_shared + cols, mask=mask, other=0.0)
            b = (b + sg * sh.to(tl.float32)).to(block_ptr.dtype.element_ty).to(tl.float32)
        g = tl.load(inject_ptr + row * stride_inject + stream).to(tl.float32)
        xr = (x + b * g).to(hc_out_ptr.dtype.element_ty)
        tl.store(hc_out_ptr + row * stride_out + offs, xr, mask=mask)
        x = xr.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / HIDDEN + eps)
    tl.store(rstd_ptr + row * HC_COUNT + stream, rstd)


class Qwen4ExpHcInjectRstd(OpWrapper):
    """With `C = hc_count`, `H = width // C`, every tensor 2-D `[T, ...]`::

        hc_out[t, s*H + i] = hyper_input[t, s*H + i] + block_out[t, i] * inject[t, s]
        rstd[t, s]         = 1 / sqrt(mean_i(hc_out[t, s*H + i]^2) + eps)

    `hc_out` is rounded once to the activation dtype and rstd is computed from that
    rounded value in fp32. Returns `(hc_out, rstd)`: `hc_out` a fresh `[T, C*H]` in
    `hyper_input`'s dtype, rstd a fresh contiguous `[T, C]` fp32. A site's normalized
    stream is `bf16(hc_out * rstd * (1 + norm_weight))`, bitwise
    `qwen4_exp_hc_inject_norm`'s `normed` (same fp32 operations and rounding).

    `block_out=None, inject=None` is the statistic alone: `hc_out` *is* `hyper_input`
    (not a copy) and only rstd is written -- the stack's first site and the site after
    the PLE add.

    With `shared` and `gate_logits` (after a MoE block), `block_out` is the routed
    experts' output and the injected block output is
    `block_out + sigmoid(gate_logits) * shared` in fp32, rounded once to `block_out`'s
    dtype (`qwen4_exp_hc_inject_norm`'s combine). None of the inputs is written.

    Fusion boundary: the shared-expert combine (optional), the stream-broadcast gated
    accumulate and the per-(row, stream) RMS statistic. The norm weight is not an
    argument: the consumers apply it.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, inject (True), moe (False: shared + gate_logits [T, 1] too). The
    # checkpoint's geometry throughout: hc_count 4, hidden 2560, bf16, contiguous rows.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MoE combine, prefill chunk 16384 (the mixed step's size)",
            spec=dict(rows=16384, moe=True),
        ),
        Cell(why="injection after attention / GDN, prefill chunk 16384", spec=dict(rows=16384)),
        Cell(why="MoE combine, prefill 2048", spec=dict(rows=2048, moe=True)),
        Cell(why="injection after attention / GDN, prefill 2048", spec=dict(rows=2048)),
        Cell(
            why="statistic only (stack start / after PLE), ragged 16390 rows",
            spec=dict(rows=16390, inject=False),
        ),
    )

    note = """
    Metadata consumed: none; every (row, stream) independent. Call sites: the trunk
    sites of `PrefillTarget` at or above `_HC_RSTD_MIN_ROWS` rows, in place of
    `qwen4_exp_hc_inject_norm`: once after each block (attention / GDN and MoE) of
    each layer, plus the statistic-only calls at the stack's first site and after
    the PLE add. Their mixers are `qwen4_exp_hc_first_rescale_tc` and
    `qwen4_exp_hc_up_gate_mean_rstd(_fp4)`, which take the site's norm weight.

    Preconditions: 2-D operands, last-dim stride 1 (row strides free); the width
    divisible by `hc_count`; `inject` `[T, hc_count]`, `block_out` and `shared`
    `[T, hidden]` of one dtype; `gate_logits` `T` elements along its first axis.
    `shared` and `gate_logits` come together and only with `block_out`.

    Numerics: `hc_out` and rstd are bitwise `qwen4_exp_hc_inject_norm`'s at the same
    row count (that kernel's two-warp form from 512 rows, eight below); rstd is the
    fp32 `rsqrt` of the fp32 mean of squares. Graph-capturable (no host sync, outputs
    allocated per call).

    Launch: one program per (row, stream), stream-minor in a 1-D grid. GB300, 16384
    rows: 0.75 GB (0.84 GB with the combine) at about 7 TB/s, against
    `qwen4_exp_hc_inject_norm`'s 1.09 / 1.17 GB.

    Certified: bf16, the checkpoint's 4 x 2560: the combine and the injection at 2048
    and 16384 rows, the statistic alone at 16390 rows; `hc_out` and the rebuilt
    normed stream bitwise `qwen4_exp_hc_inject_norm`'s; graph replay.

    Not certified: hc_count other than 4, hidden other than 2560, float16, the
    finalize fold (`topk_rows`; the prefill sites do not fold), zero rows.

    Hard limits: row offsets are int64; the grid is `rows * hc_count` programs along
    x (2^31 - 1).
    """

    def __call__(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        eps: float,
        hc_count: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows, width = hyper_input.shape
        hidden = width // hc_count
        rstd = torch.empty((rows, hc_count), dtype=torch.float32, device=hyper_input.device)
        has_inject = block_out is not None
        has_shared = shared is not None
        hc_out = torch.empty_like(hyper_input) if has_inject else hyper_input
        if rows > 0:
            block = triton.next_power_of_2(hidden)
            _hc_inject_rstd_kernel[(rows * hc_count,)](
                hyper_input,
                block_out if has_inject else hyper_input,
                inject if has_inject else hyper_input,
                shared if has_shared else hyper_input,
                gate_logits if has_shared else hyper_input,
                hc_out,
                rstd,
                hyper_input.stride(0),
                block_out.stride(0) if has_inject else 0,
                inject.stride(0) if has_inject else 0,
                shared.stride(0) if has_shared else 0,
                gate_logits.stride(0) if has_shared else 0,
                hc_out.stride(0),
                eps,
                HC_COUNT=hc_count,
                HIDDEN=hidden,
                BLOCK=block,
                HAS_INJECT=has_inject,
                HAS_SHARED=has_shared,
                num_warps=(2 if rows >= _FEW_WARPS_MIN_ROWS else 8) if block >= 2048 else 4,
            )
        return hc_out, rstd

    def reference(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        eps: float,
        hc_count: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """float64: the combine (with `shared`) rounded once to `block_out`'s dtype,
        `hc_out` the accumulate rounded once, rstd the float64 statistic of that
        rounded residual returned in fp32 -- the dataflow the op states.
        """
        dtype = hyper_input.dtype
        rows, width = hyper_input.shape
        hidden = width // hc_count
        if block_out is None:
            hc_out = hyper_input
        else:
            if shared is not None:
                s = torch.sigmoid(gate_logits.double().reshape(rows, 1))
                block_out = (block_out.double() + s * shared.double()).to(block_out.dtype)
            streams = hyper_input.double().unflatten(-1, (hc_count, hidden))
            gated = block_out.double().unsqueeze(-2) * inject.double().unsqueeze(-1)
            hc_out = (streams + gated).flatten(-2).to(dtype)
        x = hc_out.double().unflatten(-1, (hc_count, hidden))
        rstd = torch.rsqrt(x.pow(2).mean(dim=-1) + eps).float()
        return hc_out, rstd

    def is_valid(
        self,
        hyper_input: torch.Tensor,
        block_out: Optional[torch.Tensor],
        inject: Optional[torch.Tensor],
        eps: float,
        hc_count: int,
        shared: Optional[torch.Tensor] = None,
        gate_logits: Optional[torch.Tensor] = None,
    ) -> None:
        # The kernel indexes with raw row strides and unit column stride, and reads
        # `inject` / `block_out` / `shared` / the gate per row: a broadcast (stride-0) or
        # a wrong-width operand would be read out of bounds or silently reused.
        assert hyper_input.dim() == 2 and hyper_input.stride(1) == 1, "hyper_input [T, width]"
        assert (block_out is None) == (inject is None), "block_out and inject come together"
        assert (shared is None) == (gate_logits is None), "shared and gate_logits come together"
        assert shared is None or block_out is not None, "the combine needs block_out"
        rows, width = hyper_input.shape
        assert width % hc_count == 0, f"width {width} not divisible by hc_count {hc_count}"
        if block_out is not None:
            assert block_out.shape == (rows, width // hc_count) and block_out.stride(1) == 1
            assert inject.shape == (rows, hc_count) and inject.stride(1) == 1
        if shared is not None:
            assert shared.shape == (rows, width // hc_count) and shared.stride(1) == 1
            assert shared.dtype == block_out.dtype, "shared in block_out's dtype"
            assert gate_logits.shape in ((rows,), (rows, 1)), "gate_logits [T, 1] or [T]"
            assert rows <= 1 or gate_logits.stride(0) != 0, "one gate logit per row"


qwen4_exp_hc_inject_rstd = Qwen4ExpHcInjectRstd()
