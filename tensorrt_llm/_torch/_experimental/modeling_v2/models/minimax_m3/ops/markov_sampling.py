# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NATIVE_MARKOV_SAMPLING: the DSpark Markov-block sampler, Gumbel-max with one fused draw per
position, as the CUDA C++ ops ``mv2_minimax_m3::minimax_m31_markov_pick`` / ``_draw`` / ``_proposal``. Bit for bit the
Triton kernels of the former ``speculative/markov_sampling.py`` on the main class (vocab % 16 == 0 with 65..128 column
chunks of 2048: tokens, biased logits, statistics and the stored proposals), which this module
replaces; every other vocabulary runs the same kernels (more than 128 chunks: each thread folds its chunks first),
distribution-exact, not bitwise vs Triton.

Position k is drawn from ``q_k = softmax((base_k + bias(d_{k-1})) / T)`` as the argmax of ``(base_k + bias) / T + g``,
``g`` Gumbel noise from a Philox stream keyed by the request's draft ``(seed, offset)`` of position k, so every rank and
every graph replay draws the same token. Served launch order: the anchor pick, then per position the Markov GEMM
(``F.linear``, torch) and the draw with the fused pick (the row's last chunk CTA picks the token and gathers its w1
row), then one proposal launch for every ``q_k``; no PDL (the GEMM between draws does not take part in it).

``sample_markov_block`` returns None only for invalid operands (``outside_contract``: dtype, device, rank, layout) or
with ``TRTLLM_M31_DISABLE_NATIVE_MARKOV_SAMPLING=1`` (read at import); the caller then runs the stock per-position
loop.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.m31_switches import (
    m31_disabled,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

NATIVE_MARKOV_SAMPLING = not m31_disabled("NATIVE_MARKOV_SAMPLING")
BLOCK = 2048
_FLOAT = (torch.float32, torch.bfloat16)
_OPS: list = []


def ops():
    """(pick, draw, proposal) when this build carries them and NATIVE_MARKOV_SAMPLING is on, else None."""
    if not _OPS:
        ops = family_ops()
        found = (
            getattr(ops, "minimax_m31_markov_pick", None),
            getattr(ops, "minimax_m31_markov_draw", None),
            getattr(ops, "minimax_m31_markov_proposal", None),
        )
        _OPS.append(found if NATIVE_MARKOV_SAMPLING and all(found) else None)
    return _OPS[0]


def outside_contract(
    logits: torch.Tensor,
    anchors: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    draft_probs: Optional[torch.Tensor],
    slot_ids: torch.Tensor,
) -> Optional[str]:
    """Why the ops do not take these operands, or None: contiguous bf16 / fp32 CUDA logits [gens, K, vocab >= 1]; bf16 /
    fp32 w1 / w2 of one dtype; int32 / int64 anchors; int64 seeds, offsets and slot ids; fp32 temperatures; an fp32
    3-D draft_probs with a unit last stride, or none."""
    if ops() is None:
        return "off (TRTLLM_M31_DISABLE_NATIVE_MARKOV_SAMPLING=1) or the ops are not built"
    if (
        logits.dim() != 3
        or logits.dtype not in _FLOAT
        or not logits.is_cuda
        or not logits.is_contiguous()
    ):
        return f"logits {logits.dtype} {tuple(logits.shape)}"
    vocab = logits.shape[2]
    if not 1 <= vocab < 2**31 or (vocab + BLOCK - 1) // BLOCK > 65535:
        return f"vocab {vocab}"
    if (
        markov_w1.dim() != 2
        or markov_w2.dim() != 2
        or markov_w1.dtype not in _FLOAT
        or markov_w2.dtype != markov_w1.dtype
        or markov_w1.stride(1) != 1
    ):
        return f"markov_w1 {markov_w1.dtype} / markov_w2 {markov_w2.dtype}"
    if anchors.dtype not in (torch.int32, torch.int64):
        return f"anchors {anchors.dtype}"
    if any(t.dtype != torch.int64 for t in (seeds, offsets, slot_ids)):
        return f"seeds {seeds.dtype} / offsets {offsets.dtype} / slots {slot_ids.dtype}"
    if temperatures.dtype != torch.float32:
        return f"temperatures {temperatures.dtype}"
    if draft_probs is not None and (
        draft_probs.dtype != torch.float32 or draft_probs.dim() != 3 or draft_probs.stride(2) != 1
    ):
        return f"draft_probs {draft_probs.dtype} {tuple(draft_probs.shape)}"
    return None


def sample_markov_block(
    logits: torch.Tensor,
    anchors: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    temperatures: torch.Tensor,
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    offset_step: int,
    draft_probs: Optional[torch.Tensor],
    slot_ids: torch.Tensor,
    padding_slot: int,
) -> Optional[torch.Tensor]:
    """Draw the ``[num_gens, K]`` int32 block from contiguous ``[num_gens, K, vocab]`` base ``logits`` (read only), or
    None outside the contract. Position k draws at Philox offset ``offsets + k * offset_step``; ``q_k`` goes to
    ``draft_probs[slot_ids, k, :vocab]`` unless ``draft_probs`` is None. No host synchronization."""
    if (
        outside_contract(
            logits,
            anchors,
            markov_w1,
            markov_w2,
            temperatures,
            seeds,
            offsets,
            draft_probs,
            slot_ids,
        )
        is not None
    ):
        return None
    pick, draw, proposal = ops()
    num_gens, num_pos, vocab = logits.shape
    rank = markov_w1.shape[1]
    num_chunks = (vocab + BLOCK - 1) // BLOCK
    device = logits.device
    tokens = torch.empty(num_gens, num_pos, dtype=torch.int32, device=device)
    emb = torch.empty(num_gens, rank, dtype=markov_w1.dtype, device=device)
    keys = torch.empty(num_gens, num_chunks, dtype=torch.float32, device=device)
    indices = torch.empty(num_gens, num_chunks, dtype=torch.int32, device=device)
    run_max = torch.empty(num_gens, num_pos, num_chunks, dtype=torch.float32, device=device)
    run_sum = torch.empty_like(run_max)
    # The biased logits the draws scale, kept in float32 for the proposal kernel to recompute exactly.
    biased = torch.empty(num_gens, num_pos, vocab, dtype=torch.float32, device=device)
    # The fused pick's per-row arrival counters: zero here, reset by each row's last chunk CTA.
    arrivals = torch.zeros(num_gens, dtype=torch.int32, device=device)
    pick(
        keys,
        indices,
        tokens,
        anchors.contiguous(),
        0,
        num_pos,
        markov_w1,
        emb,
        vocab,
        num_chunks,
        False,
        False,
    )
    for k in range(num_pos):
        bias = F.linear(emb, markov_w2)
        draw(
            logits, biased, bias, temperatures, seeds, offsets, slot_ids, keys, indices, run_max, run_sum,
            k, num_pos, offset_step, padding_slot, vocab, num_chunks, arrivals, tokens, markov_w1, emb, False,
        )  # fmt: skip
    if draft_probs is not None:
        proposal(
            biased,
            temperatures,
            run_max,
            run_sum,
            slot_ids,
            draft_probs,
            vocab,
            num_chunks,
            num_pos,
            False,
        )
    return tokens
