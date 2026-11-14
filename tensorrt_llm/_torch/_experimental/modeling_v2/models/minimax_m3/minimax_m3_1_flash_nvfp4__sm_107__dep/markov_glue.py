# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_MARKOV_GLUE: the DSpark ancestral draft loop's per-position glue (``DSparkWorker._sample_block_ancestrally``)
in three launches a position, through the ancestral glue override (``speculative/dspark.set_ancestral_glue_override``).

Around the Markov w2 GEMV and the sampler, each of the K draft positions runs the stock lines
  bias = dspark_markov_step_bias(tokens, w1, w2)     the masked w1 gather (get_masked_input_and_mask, F.embedding,
                                                     masked_fill: ~11 launches), then F.linear
  x    = gen_logits[:, step] + bias.to(float32)      a widening copy of the bias, then a strided add
  draft_probs[slots, step, :vocab] = probs           an index_put over the slot rows
``step_bias`` gathers with the catalog's ``dspark_markov_bias.prev_embeddings`` (one launch: w1's rows bit for bit, and
a +0.0 row for an id outside [0, V), as masked_fill(0)) and keeps the stock F.linear. ``biased_logits`` widens the bias
and adds it to the strided fp32 base in one launch, the family CUDA C++ op
``mv2_minimax_m3::minimax_m31_widen_add_rows`` (``minimax_m3/ops/row_ops``): one IEEE fp32 add of two exactly widened
values, as torch's. ``store_probs`` is the in-tree ``mv2_minimax_m3::minimax_m31_scatter_rows`` row copy on the step's
view of ``draft_probs``. The bias, the logits, the sampled tokens and the stored proposals are therefore the stock ones
bit for bit. The CUDA-graph padding requests share one scratch slot, which no real request reads; their draws differ
(the sampler's Philox subsequence is the row), so with either store that slot ends with each element from one of their
proposals.

Contract: each method's ``*_outside_contract`` names the layouts it takes; anything else returns None (False for
``store_probs``), logs a "not taken" line once per reason, and the stock line runs. Kill switches:
``TRTLLM_M31_DISABLE_DEP_MARKOV_GLUE=1`` installs no override; ``TRTLLM_M31_DISABLE_NATIVE_WIDEN_ADD=1`` keeps the
override but declines ``biased_logits``, so the stock ``gen_logits[:, step] + bias.to(float32)`` runs. Read at
import.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_markov_bias,
)
from tensorrt_llm.logger import logger

from ..ops import row_ops
from .m31_switches import m31_disabled

DEP_MARKOV_GLUE = not m31_disabled("DEP_MARKOV_GLUE")
NATIVE_WIDEN_ADD = not m31_disabled("NATIVE_WIDEN_ADD")

_INT = (torch.int32, torch.int64)
_BIAS = (torch.bfloat16, torch.float32)
_STATE: dict = {}
_TAKEN: set = set()
_NOT_TAKEN: set = set()


def scatter_rows_op():
    """``mv2_minimax_m3::minimax_m31_scatter_rows`` when this build carries it, else None."""
    if "scatter" not in _STATE:
        _STATE["scatter"] = getattr(family_ops(), "minimax_m31_scatter_rows", None)
    return _STATE["scatter"]


def bias_outside_contract(prev_tokens: torch.Tensor, markov_w1: torch.Tensor) -> Optional[str]:
    """Why the gather does not take these operands, or None: a contiguous 1-D int32 / int64 CUDA ``prev_tokens`` with
    at least one entry, and a contiguous 2-D bf16 ``markov_w1`` on its device whose rank is a power of two in
    [16, 1024] (``dspark_markov_bias.prev_embeddings``)."""
    if not isinstance(prev_tokens, torch.Tensor) or not isinstance(markov_w1, torch.Tensor):
        return "not tensors"
    if not prev_tokens.is_cuda or prev_tokens.dim() != 1 or prev_tokens.dtype not in _INT:
        return f"prev_tokens {prev_tokens.dtype} {tuple(prev_tokens.shape)} on {prev_tokens.device}"
    if prev_tokens.numel() < 1 or not prev_tokens.is_contiguous():
        return f"prev_tokens {tuple(prev_tokens.shape)} empty or strided"
    rank = markov_w1.shape[-1] if markov_w1.dim() == 2 else 0
    if (
        markov_w1.dim() != 2
        or markov_w1.dtype != torch.bfloat16
        or not markov_w1.is_contiguous()
        or markov_w1.device != prev_tokens.device
        or not 16 <= rank <= 1024
        or rank & (rank - 1)
    ):
        return f"markov_w1 {markov_w1.dtype} {tuple(markov_w1.shape)} on {markov_w1.device}"
    if not dspark_markov_bias.prev_embeddings_available():
        return "minimax_m31_dspark_prev_embeddings unavailable"
    return None


def logits_outside_contract(
    gen_logits: torch.Tensor, step: int, bias: torch.Tensor
) -> Optional[str]:
    """Why the widening add does not take these operands, or None: a 3-D float32 CUDA ``gen_logits`` [rows, K, vocab]
    with a unit inner stride, ``0 <= step < K``, at most 65535 rows, and a contiguous bf16 / float32 ``bias``
    [rows, vocab] on its device; NATIVE_WIDEN_ADD on and ``mv2_minimax_m3::minimax_m31_widen_add_rows`` built."""
    if not NATIVE_WIDEN_ADD:
        return "off (TRTLLM_M31_DISABLE_NATIVE_WIDEN_ADD=1)"
    if row_ops.widen_add_rows_op() is None:
        return "mv2_minimax_m3::minimax_m31_widen_add_rows is not built"
    if not isinstance(gen_logits, torch.Tensor) or not isinstance(bias, torch.Tensor):
        return "not tensors"
    if (
        not gen_logits.is_cuda
        or gen_logits.dtype != torch.float32
        or gen_logits.dim() != 3
        or gen_logits.stride(2) != 1
    ):
        return (
            f"gen_logits {gen_logits.dtype} {tuple(gen_logits.shape)} strides {gen_logits.stride()}"
        )
    rows, k, vocab = gen_logits.shape
    if not 0 <= step < k or not 1 <= rows <= 65535 or vocab < 1:
        return f"step {step} of gen_logits {tuple(gen_logits.shape)}"
    if (
        bias.dtype not in _BIAS
        or tuple(bias.shape) != (rows, vocab)
        or not bias.is_contiguous()
        or bias.device != gen_logits.device
    ):
        return f"bias {bias.dtype} {tuple(bias.shape)} for gen_logits {tuple(gen_logits.shape)}"
    return None


def store_outside_contract(
    draft_probs: torch.Tensor, slots: torch.Tensor, step: int, probs: torch.Tensor
) -> Optional[str]:
    """Why the row copy does not take these operands, or None: a float32 CUDA ``draft_probs`` [slots, K, >= vocab] with
    a unit inner stride, ``0 <= step < K``, a contiguous float32 ``probs`` [rows, vocab] and a contiguous int64
    ``slots`` [rows] on its device (``mv2_minimax_m3::minimax_m31_scatter_rows``)."""
    if scatter_rows_op() is None:
        return "mv2_minimax_m3::minimax_m31_scatter_rows is not built"
    if not all(isinstance(t, torch.Tensor) for t in (draft_probs, slots, probs)):
        return "not tensors"
    if (
        not draft_probs.is_cuda
        or draft_probs.dtype != torch.float32
        or draft_probs.dim() != 3
        or draft_probs.stride(2) != 1
    ):
        return f"draft_probs {draft_probs.dtype} {tuple(draft_probs.shape)} strides {draft_probs.stride()}"
    if probs.dtype != torch.float32 or probs.dim() != 2 or not probs.is_contiguous():
        return f"probs {probs.dtype} {tuple(probs.shape)} strided"
    rows, vocab = probs.shape
    if (
        not 0 <= step < draft_probs.shape[1]
        or vocab > draft_probs.shape[2]
        or not 1 <= rows <= 65535
    ):
        return (
            f"step {step} / probs {tuple(probs.shape)} for draft_probs {tuple(draft_probs.shape)}"
        )
    if (
        slots.dtype != torch.int64
        or slots.dim() != 1
        or slots.numel() != rows
        or not slots.is_contiguous()
        or not draft_probs.device == probs.device == slots.device
    ):
        return f"slots {slots.dtype} {tuple(slots.shape)} for {rows} rows"
    return None


def _taken(method: str, rows: int) -> None:
    """The taken line: once per method."""
    if method in _TAKEN:
        return
    _TAKEN.add(method)
    logger.info(f"[M3.1 DEP] DEP_MARKOV_GLUE {method} taken (at {rows} rows)")


def _not_taken(method: str, reason: str) -> None:
    """The not-taken line: once per method and reason."""
    if (method, reason) in _NOT_TAKEN:
        return
    _NOT_TAKEN.add((method, reason))
    logger.info(f"[M3.1 DEP] DEP_MARKOV_GLUE {method} not taken ({reason}); the stock line runs")


class MarkovGlue:
    """The ancestral glue override (see the module docstring); each method declines to the stock line."""

    @staticmethod
    def step_bias(
        prev_tokens: torch.Tensor, markov_w1: torch.Tensor, markov_w2: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """``dspark_markov_step_bias(prev_tokens, markov_w1, markov_w2)`` bit for bit, or None."""
        reason = bias_outside_contract(prev_tokens, markov_w1)
        if reason is not None:
            _not_taken("step_bias", reason)
            return None
        emb = dspark_markov_bias.prev_embeddings(prev_tokens, prev_tokens.view(-1, 1), markov_w1)
        _taken("step_bias", prev_tokens.numel())
        return F.linear(emb, markov_w2)

    @staticmethod
    def biased_logits(
        gen_logits: torch.Tensor, step: int, bias: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """``gen_logits[:, step] + bias.to(gen_logits.dtype)`` bit for bit (a contiguous float32 [rows, vocab]), or
        None."""
        reason = logits_outside_contract(gen_logits, step, bias)
        if reason is not None:
            _not_taken("biased_logits", reason)
            return None
        out = row_ops.widen_add_rows(gen_logits[:, step], bias)
        _taken("biased_logits", gen_logits.shape[0])
        return out

    @staticmethod
    def store_probs(
        draft_probs: torch.Tensor, slots: torch.Tensor, step: int, probs: torch.Tensor
    ) -> bool:
        """``draft_probs[slots, step, :vocab] = probs`` bit for bit; False (nothing written) to decline."""
        reason = store_outside_contract(draft_probs, slots, step, probs)
        if reason is not None:
            _not_taken("store_probs", reason)
            return False
        rows, vocab = probs.shape
        scatter_rows_op()(draft_probs[:, step : step + 1], slots, probs.view(rows, 1, vocab))
        _taken("store_probs", rows)
        return True


def engagement() -> str:
    """The log line's state: what the override does in this process."""
    if not DEP_MARKOV_GLUE:
        return "off (TRTLLM_M31_DISABLE_DEP_MARKOV_GLUE=1); the stock gather, widening add and index_put"
    store = (
        "row copy" if scatter_rows_op() is not None else "stock index_put (scatter_rows not built)"
    )
    add = (
        "one-launch widening add"
        if NATIVE_WIDEN_ADD and row_ops.widen_add_rows_op() is not None
        else "stock widening add"
    )
    return f"engaged (one-launch gather, {add}, {store})"


def log_engagement() -> None:
    logger.info(f"[M3.1 DEP] DEP_MARKOV_GLUE: {engagement()}")
