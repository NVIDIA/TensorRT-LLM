# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The full-attention layer's split merge, folded into the attention norm-gate (PDL launch).

``full_attention_split_kv`` leaves one partial per (row, split): the normalized attention output in FP16
and its log2-sum-exp. This entry is ``norm_gate``'s ``_attention_norm_gate_pdl_kernel`` -- one
Triton program per (row, head), the same norm and sigmoid gate statement for statement -- whose
attention operand is the merge of those partials instead of a BF16 attention output read from
memory: each program loads its row's log2-sum-exp and O of every keyed split at once, weights
split ``s`` by ``2^(lse_s - max lse)`` (a split with ``-inf`` gets 0 and its O is not used),
sums in a fixed reduction order (deterministic), divides, rounds to BF16 (the attention output's
dtype on the built-in path), and then runs the unchanged norm-gate. ``combine_only`` stores that
BF16 attention output instead (tests, and anything that wants the merged attention alone).

Which splits are read: given the step's ``seq_lens`` (the same tensor ``full_attention_split_kv`` read),
the kernel derives the keyed split count ``min(splits, ceil(ceil(L / 32) / MIN_PAGES))`` as
``full_attention_split_kv`` does and loads neither the O nor the log2-sum-exp of the key-less splits past
it; without ``seq_lens`` it loads every split's and relies on their ``-inf`` alone. Both give
the same bits on ``full_attention_split_kv``'s partials.

PDL memory order: before ``griddepcontrol.wait`` the kernel reads only ``seq_lens[0]`` (when
given), which no kernel of the step writes (the same record as ``full_attention_split_kv``'s); every
partial and gate read follows the wait -- the partials are the previous kernel's. It triggers
its dependent as ``norm_gate`` does (``trigger``).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

from .full_attention_split_kv import MIN_PAGES, PAGE

#: Width of one attention head.
HEAD_DIM = 64
#: Query heads of one tep8 rank (the default ``heads``).
HEADS = 16
#: Query heads of one rank per tensor-parallel size: what ``heads`` may be.
RANK_HEADS = {8: HEADS, 4: 2 * HEADS}
#: Rows of a decode step: one request's last token and 7 drafts.
MAX_ROWS = 8
#: Sequences one call merges (``requests``): ``full_attention_split_kv_multi``'s partials.
MAX_REQUESTS = 4
#: Largest number of splits the kernel reads.
MAX_SPLITS = 512

EXIT, AFTER_WAIT, ENTRY = 0, 1, 2
TRIGGERS = {"exit": EXIT, "after_wait": AFTER_WAIT, "entry": ENTRY}


@triton.jit
def _combine_norm_gate_pdl_kernel(
    out,
    part_o,
    part_lse,
    gate,
    seq_lens,
    splits,
    gate_token_stride,
    gate_head_stride,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SPLITS_P2: tl.constexpr,
    TRIGGER: tl.constexpr,
    COMBINE_ONLY: tl.constexpr,
    KEYED_BY_LENGTH: tl.constexpr,
    PAGE: tl.constexpr,
    MIN_PAGES: tl.constexpr,
    REQS: tl.constexpr = 1,
    ROWS_PER_REQ: tl.constexpr = 8,
):
    if TRIGGER == 2:
        tl.extra.cuda.gdc_launch_dependents()
    keyed = splits
    if REQS > 1:
        # Several sequences (full_attention_split_kv_multi): this row's sequence, its splits
        # (request_of: at least one each, the rest in proportion to the pages; every length is
        # a pre-wait read as below) and its keyed ones, and its row of the partials.
        req_token = tl.program_id(0) // HEADS
        req = req_token // ROWS_PER_REQ
        total = 0
        before = 0
        mine = 0
        for i in tl.static_range(REQS):
            n_i = (tl.load(seq_lens + i) + PAGE - 1) // PAGE
            total += n_i
            before += tl.where(i < req, n_i, 0)
            mine += tl.where(i == req, n_i, 0)
        lo = req + (splits - REQS) * before // total
        hi = tl.where(req + 1 < REQS, req + 1 + (splits - REQS) * (before + mine) // total, splits)
        keyed = tl.minimum(hi - lo, (mine + MIN_PAGES - 1) // MIN_PAGES)
        # A sequence's block of partial rows is 8 tokens (MAX_ROWS) x HEADS.
        part_row = req * (8 * HEADS) + (req_token % ROWS_PER_REQ) * HEADS + tl.program_id(0) % HEADS
    elif KEYED_BY_LENGTH:
        # Before the wait: the context length, which no kernel of the step writes. The keyed
        # splits are the first min(splits, ceil(pages / MIN_PAGES)), as full_attention_split_kv cuts them.
        pages = (tl.load(seq_lens) + PAGE - 1) // PAGE
        keyed = tl.minimum(splits, (pages + MIN_PAGES - 1) // MIN_PAGES)
    # Before any other global read: the partials are the previous kernel's.
    tl.extra.cuda.gdc_wait()
    if TRIGGER == 1:
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    if REQS == 1:
        part_row = row
    offsets = tl.arange(0, HEAD_DIM)
    s = tl.arange(0, SPLITS_P2)
    in_range = s < keyed
    # Every keyed split's log2-sum-exp and O of this row, issued together.
    lse = tl.load(part_lse + part_row * splits + s, mask=in_range, other=float("-inf"))
    o = tl.load(
        part_o + (part_row * splits + s)[:, None] * HEAD_DIM + offsets[None, :],
        mask=in_range[:, None],
        other=0.0,
    ).to(tl.float32)
    valid = lse > float("-inf")
    m = tl.max(lse, axis=0)
    w = tl.where(valid, tl.exp2(lse - m), 0.0)
    o = tl.where(valid[:, None], o, 0.0)
    merged = tl.sum(w[:, None] * o, axis=0) / tl.sum(w, axis=0)
    if COMBINE_ONLY:
        tl.store(out + row * HEAD_DIM + offsets, merged.to(tl.bfloat16))
    else:
        # From here on: the built-in _attention_norm_gate_kernel on the BF16 attention output.
        x = merged.to(tl.bfloat16).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / HEAD_DIM
        normed = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
        g = tl.load(gate + token * gate_token_stride + head * gate_head_stride + offsets).to(
            tl.float32
        )
        sigmoid = tl.sigmoid(g).to(tl.bfloat16)
        result = normed.to(tl.float32) * sigmoid.to(tl.float32)
        tl.store(out + row * HEAD_DIM + offsets, result)


def gate_head_stride(gate: torch.Tensor) -> int:
    """The stride between heads of ``gate`` as the built-in launch computes it."""
    return gate.stride(1) if gate.ndim == 3 else HEAD_DIM


def _launch(
    out,
    part_o,
    part_lse,
    rows,
    gate,
    eps,
    trigger,
    combine_only,
    seq_lens=None,
    heads=HEADS,
    requests=1,
):
    splits = part_lse.shape[1]
    _combine_norm_gate_pdl_kernel[(rows * heads,)](
        out,
        part_o,
        part_lse,
        gate if gate is not None else out,
        seq_lens if seq_lens is not None else part_lse,
        splits,
        gate.stride(0) if gate is not None else 0,
        gate_head_stride(gate) if gate is not None else 0,
        eps,
        HEADS=heads,
        HEAD_DIM=HEAD_DIM,
        SPLITS_P2=triton.next_power_of_2(splits),
        TRIGGER=TRIGGERS[trigger],
        COMBINE_ONLY=combine_only,
        KEYED_BY_LENGTH=seq_lens is not None,
        PAGE=PAGE,
        MIN_PAGES=MIN_PAGES,
        num_warps=4,
        launch_pdl=True,
        # Passed only for several sequences: one sequence launches the kernel without them.
        **({"REQS": requests, "ROWS_PER_REQ": rows // requests} if requests > 1 else {}),
    )
    return out


def _checks(part_o, part_lse, rows, gate, trigger, out, seq_lens=None, heads=HEADS, requests=1):
    splits = part_lse.shape[1] if part_lse.dim() == 2 else 0
    per_req = rows // requests if 1 <= requests <= MAX_REQUESTS else 0
    bad = (
        heads not in RANK_HEADS.values()
        or not 1 <= requests <= MAX_REQUESTS
        or rows % max(requests, 1) != 0
        or (requests > 1 and (seq_lens is None or splits <= requests))
        or part_o.dtype != torch.float16
        or part_o.dim() != 3
        or part_o.shape[0] < (rows * heads if requests == 1 else requests * MAX_ROWS * heads)
        or part_o.shape[2] != HEAD_DIM
        or not part_o.is_contiguous()
        or part_lse.dtype != torch.float32
        or part_lse.dim() != 2
        or not part_lse.is_contiguous()
        or part_o.shape[:2] != part_lse.shape
        or not 1 <= splits <= MAX_SPLITS
        or not 1 <= per_req <= MAX_ROWS
        or trigger not in TRIGGERS
    )
    if gate is not None:
        bad = bad or (
            gate.dtype != torch.bfloat16
            or gate.ndim not in (2, 3)
            or gate.stride(-1) != 1
            or gate.shape[0] != rows
            or gate.numel() != rows * heads * HEAD_DIM
            or (gate.ndim == 3 and gate.shape[-1] != HEAD_DIM)
        )
    if out is not None:
        bad = bad or (
            out.dtype != torch.bfloat16
            or not out.is_contiguous()
            or tuple(out.shape) != (rows, heads * HEAD_DIM)
        )
    if seq_lens is not None:
        bad = bad or (
            seq_lens.dtype != torch.int32
            or seq_lens.numel() < max(requests, 1)
            or seq_lens.device != part_o.device
        )
    if bad:
        raise ValueError(
            f"combine_norm_gate takes H query heads in {sorted(RANK_HEADS.values())}, 1.."
            f"{MAX_REQUESTS} requests (more than one: with seq_lens, more splits than requests, "
            f"partials of [requests * {MAX_ROWS} * H, ...]), contiguous "
            f"partials FP16 [>= rows * H, splits, "
            f"{HEAD_DIM}] and fp32 [same, splits] (1..{MAX_SPLITS} splits), 1..{MAX_ROWS} rows, "
            f"a BF16 gate [rows, H * {HEAD_DIM}] or [rows, H, {HEAD_DIM}] with unit "
            f"last stride, a trigger of {sorted(TRIGGERS)} and optional int32 seq_lens on the "
            f"partials' device; got H {heads}, part_o {part_o.dtype} "
            f"{tuple(part_o.shape)}, part_lse {part_lse.dtype} {tuple(part_lse.shape)}, rows "
            f"{rows}, gate {None if gate is None else (gate.dtype, tuple(gate.shape))}, "
            f"trigger {trigger!r}, requests {requests}, seq_lens "
            f"{None if seq_lens is None else (seq_lens.dtype, tuple(seq_lens.shape))}"
        )


class CombineNormGate(OpWrapper):
    """``bf16(rmsnorm(merge(partials)))`` times ``bf16(sigmoid(gate))`` per (row, head), PDL.

    ``heads`` (H) is the rank's query heads: 16 at tep8 (the default), 32 at tep4.
    ``part_o`` ``[>= rows * H, splits, 64]`` FP16 and ``part_lse`` ``[same, splits]`` fp32
    are ``full_attention_split_kv``'s partials (row = token * H + query head); ``gate`` is the gate
    ``[rows, H * 64]`` or ``[rows, H, 64]`` BF16 with unit last stride. Returns a fresh
    contiguous ``[rows, H * 64]`` BF16 tensor. ``trigger`` names when the dependent may launch;
    it changes nothing the kernel computes. ``seq_lens`` (the int32 tensor ``full_attention_split_kv`` read;
    optional) lets the kernel skip the key-less splits' loads; it changes no bit when the
    partials are ``full_attention_split_kv``'s of that context.

    ``requests`` (R, 2..4, with ``seq_lens``): ``full_attention_split_kv_multi``'s partials of R
    sequences of ``rows / R`` rows each (sequence ``r``'s rows at ``r * 8 H + token * H + head``,
    its ``request_splits`` count of splits first on the split axis); output and gate rows are
    the step's rows in order.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    The norm-gate half is norm_gate's kernel statement for statement on the BF16-rounded merge,
    certified bit for bit against attention_norm_gate applied to combine_only's output. The
    merge is gated against merge_partials (fp64) within a derived band, with key-less splits
    whose O holds NaN bytes, and is deterministic. Rows 1..8. With seq_lens the splits past the
    keyed count are not read (garbage there changes nothing), and the result is bit for bit the
    unmasked one on full_attention_split_kv's partials.
    """

    def __call__(
        self,
        part_o: torch.Tensor,
        part_lse: torch.Tensor,
        rows: int,
        gate: torch.Tensor,
        eps: float,
        trigger: str = "after_wait",
        out: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        heads: int = HEADS,
        requests: int = 1,
    ) -> torch.Tensor:
        _checks(part_o, part_lse, rows, gate, trigger, out, seq_lens, heads, requests)
        if out is None:
            out = torch.empty(rows, heads * HEAD_DIM, dtype=torch.bfloat16, device=part_o.device)
        return _launch(
            out, part_o, part_lse, rows, gate, eps, trigger, False, seq_lens, heads, requests
        )

    def reference(
        self,
        part_o: torch.Tensor,
        part_lse: torch.Tensor,
        rows: int,
        gate: torch.Tensor,
        eps: float,
        trigger: str = "after_wait",
        out: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        heads: int = HEADS,
        requests: int = 1,
    ) -> torch.Tensor:
        """fp64 merge, rounded to BF16 as the op does, then fp64 between the norm-gate's BF16
        rounding points. ``trigger`` and ``out`` change when and where, not what; with
        ``seq_lens`` only the keyed splits are merged; several ``requests`` are each merged
        over their own splits (``request_splits``)."""
        from .full_attention_split_kv import keyed_splits, merge_partials, request_splits

        if requests > 1:
            per_req = rows // requests
            block = MAX_ROWS * heads
            spans = request_splits(seq_lens[:requests].tolist(), part_lse.shape[1])
            return torch.cat(
                [
                    self.reference(
                        part_o[r * block : (r + 1) * block, :n],
                        part_lse[r * block : (r + 1) * block, :n],
                        per_req,
                        gate[r * per_req : (r + 1) * per_req],
                        eps,
                        seq_lens=seq_lens[r : r + 1],
                        heads=heads,
                    )
                    for r, (_, n) in enumerate(spans)
                ]
            )
        if seq_lens is not None:
            keyed = keyed_splits(int(seq_lens[0]), part_lse.shape[1])
            part_o, part_lse = part_o[:, :keyed], part_lse[:, :keyed]
        x = merge_partials(part_o[: MAX_ROWS * heads], part_lse[: MAX_ROWS * heads], rows)
        x = x.double().to(torch.bfloat16).double().reshape(rows, heads, HEAD_DIM)
        normed = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        g = gate.reshape(rows, heads, HEAD_DIM).double()
        sigmoid = torch.sigmoid(g).to(torch.bfloat16)
        return (normed.double() * sigmoid.double()).to(torch.bfloat16).reshape(rows, -1)


def combine_only(
    part_o: torch.Tensor,
    part_lse: torch.Tensor,
    rows: int,
    trigger: str = "exit",
    seq_lens: Optional[torch.Tensor] = None,
    heads: int = HEADS,
    requests: int = 1,
) -> torch.Tensor:
    """The merged attention output ``[rows, H * 64]`` BF16 of ``full_attention_split_kv``'s partials
    (``seq_lens``, ``heads`` and ``requests`` as in ``combine_norm_gate``)."""
    _checks(part_o, part_lse, rows, None, trigger, None, seq_lens, heads, requests)
    out = torch.empty(rows, heads * HEAD_DIM, dtype=torch.bfloat16, device=part_o.device)
    return _launch(out, part_o, part_lse, rows, None, 0.0, trigger, True, seq_lens, heads, requests)


combine_norm_gate = CombineNormGate()
