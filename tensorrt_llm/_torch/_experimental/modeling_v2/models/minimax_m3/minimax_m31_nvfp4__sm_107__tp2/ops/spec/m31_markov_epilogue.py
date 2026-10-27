# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S4FE: the in-tree DSpark Markov chain (the steps S4's one-launch chain declines: above MAX_FAST_BATCH requests, full
vocab, replicated on both ranks) with its elementwise tail fused per position.

The in-tree chain (models/modeling_speculative.py dspark_markov_chain, cast_bias_to_logits, the unsharded argmax of
interface.py greedy_sample_draft_with_tp_gather) runs per position k: bias = F.linear(markov_prev_embeddings(prev, w1),
w2) (bf16), then bias.to(fp32), base[:, k] + bias, argmax, and finally torch.cat of the corrected rows. Here the same
embedding (position 0) and the same F.linear run; ``trtllm::minimax_m31_markov_epilogue`` does the rest in one pass:
the fp32 add (one add.rn, as torch's), the corrected row written in place of the cat, torch's argmax order (NaN wins,
ties take the lower index) and the next position's embedding w1[prev] (prev is in vocab, so the mask is a no-op).
Bitwise the in-tree corrected logits by construction.

Workspace: per device, a partials buffer (rows x splits x 8 B) and rows int32 counters that every launch leaves zero,
allocated once outside capture (``prebuild``)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.models.modeling_speculative import (
    dspark_markov_chain_logits,
    markov_prev_embeddings,
)

# Request capacity covers 448 verify rows at K3 (112 requests).
MAX_ROWS = 112
# Splits per row are at most d / 4 / 1024 (markovEpilogueSplits); the workspace holds the 200064-vocab maximum.
MAX_SPLITS = 1024

_STATE: dict = {}
_BK3_LOGGED: set = set()
_WORKSPACES: dict = {}


def markov_epilogue_op():
    """``trtllm::minimax_m31_markov_epilogue`` if this build carries it, else None."""
    if "op" not in _STATE:
        op = getattr(torch.ops.trtllm, "minimax_m31_markov_epilogue", None)
        _STATE["op"] = op
    return _STATE["op"]


def _prev_embeddings(first_prev: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    """Position 0's embedding: TAIL_BK3 (B) ``trtllm::minimax_m31_markov_prev_embed`` (one gather, bitwise the in-tree
    masked embedding) when on and in layout, else the in-tree ``markov_prev_embeddings``."""
    from ...dspark_s4 import TAIL_BK3

    op = getattr(torch.ops.trtllm, "minimax_m31_markov_prev_embed", None) if TAIL_BK3 else None
    prev = first_prev.long()
    if (
        op is not None
        and prev.dim() == 1
        and w1.dim() == 2
        and w1.dtype == torch.bfloat16
        and w1.is_contiguous()
    ):
        rows = int(prev.shape[0])
        if rows not in _BK3_LOGGED:
            _BK3_LOGGED.add(rows)
            from tensorrt_llm.logger import logger

            logger.info(f"[M3.1] TAIL_BK3 Markov embedding taken at {rows} requests")
        return op(prev.contiguous(), w1)
    return markov_prev_embeddings(prev, w1)


def prebuild(device: torch.device) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Allocate (once, outside capture) the device's (partials, counters); None when the op is not built."""
    if markov_epilogue_op() is None:
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _WORKSPACES:
        if torch.cuda.is_current_stream_capturing():
            return None
        dev = torch.device("cuda", index)
        _WORKSPACES[index] = (
            torch.empty(MAX_ROWS * MAX_SPLITS * 8, dtype=torch.uint8, device=dev),
            torch.zeros(MAX_ROWS, dtype=torch.int32, device=dev),
        )
    return _WORKSPACES[index]


def _splits(rows: int, d: int) -> int:
    key = ("splits", rows, d)
    if key not in _STATE:
        _STATE[key] = int(torch.ops.trtllm.minimax_m31_markov_epilogue_splits(rows, d))
    return _STATE[key]


class M31MarkovChainFused(OpWrapper):
    """The in-tree Markov chain's corrected logits [B, K, d] (full vocab), with the elementwise tail fused; None when
    the op is not built, the step is wider than MAX_ROWS or the operands are outside the kernel's layout (the caller
    keeps the in-tree chain)."""

    ARCHS = frozenset({Arch.SM_107})

    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"{b} requests at K{k} (agg TP2 above S4's cap)", spec=dict(b=b, k=k, d=200064))
        for k, rows in ((3, range(13, 113)), (7, range(13, 57)))
        for b in rows
    )

    note = """
    The F.linear and the position-0 embedding are the in-tree calls themselves, so cuBLAS picks the same tactic; the
    kernel's add, argmax order and embedding copy are exact, so the corrected logits are the in-tree bits.
    """

    def __call__(
        self, base: torch.Tensor, first_prev: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
    ) -> Optional[torch.Tensor]:
        op = markov_epilogue_op()
        if (
            op is None
            or base.dim() != 3
            or base.dtype != torch.float32
            or not base.is_contiguous()
            or not 1 <= base.shape[0] <= MAX_ROWS
            or base.shape[2] % 4 != 0
            or w1.dtype != torch.bfloat16
            or w2.dtype != torch.bfloat16
            or not w1.is_contiguous()
            or w2.shape[0] != base.shape[2]
            or w1.shape[0] < base.shape[2]
        ):
            return None
        ws = prebuild(base.device)
        if ws is None:
            return None
        partials, counters = ws
        rows, block, d = base.shape
        splits = _splits(rows, d)
        out = torch.empty_like(base)
        prev = torch.empty(rows, dtype=torch.int64, device=base.device)
        emb = _prev_embeddings(first_prev, w1)
        for k in range(block):
            bias = F.linear(emb, w2)
            nxt = torch.empty_like(emb) if k + 1 < block else None
            op(base, k, bias, out, prev, w1, nxt, partials, counters, splits)
            emb = nxt
        return out

    def reference(self, base, first_prev, w1, w2) -> torch.Tensor:
        """The in-tree chain with the unsharded argmax (interface.py greedy_sample_draft_with_tp_gather)."""
        return dspark_markov_chain_logits(
            base, first_prev, w1, w2,
            argmax_fn=lambda x: torch.argmax(x, dim=-1).type(torch.int32).long(),
        )  # fmt: skip

    def is_valid(self, base, first_prev, w1, w2) -> None:
        assert base.dtype == torch.float32 and base.dim() == 3, "base: fp32 [B, K, d]"


m31_markov_chain_fused = M31MarkovChainFused()
