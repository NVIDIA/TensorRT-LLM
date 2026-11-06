# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash 2 sampled path walk and its draft distributions, in one launch.

Rejection sampling drafts each block position under the token drawn before it
(``dflash2_sample_candidate_paths`` with ``DFlashWorker._sample_dflash2_block``'s
``sample_step``). Per position the built-in runs a gather of the lattice row
under the predecessor, ``nan_to_num``, the rows' Philox offsets, the shipped
fused sampler with probabilities (``fused_sample_from_logits_with_probs``: one
CTA of 1024 threads, or 512 above 128 rows, per ``top_k``-wide row) and a clamp
of the drawn index; then a gather of the drawn candidates' vocabulary ids and a
scatter of every position's distribution into the slot-indexed ``draft_probs``.

``draft_path_sample`` (``_head_sampler/draft_path_sample.cuh``) runs one CTA of
the shipped block per request through every position: it writes the position's
row with the built-in's single FP32 add and ``nan_to_num`` and the rows' offsets
into scratch shaped as the built-in's tensors, calls the shipped sampler body on
them (row ``b`` is ``blockIdx.x == b`` in both, as in the shipped launch), and
keeps the clamped index as the next predecessor. Tokens, distributions and
``draft_probs`` writes are the built-in's bit for bit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper


class DraftPathSample(OpWrapper):
    """Sampled DFlash 2 draft tokens from the selector's lattice.

        edges          [B, K, top_k, top_k] fp32: the einsum ``bilinear[b, t, pred, cand]``
                       (without the unary term), contiguous
        unary          [B, K, top_k] fp32, contiguous
        candidate_ids  [B, K, top_k] int64 vocab ids, contiguous
        temperatures, top_ks, top_ps, min_ps
                       per-request sampling parameters ([>= B]; fp32, int32, fp32, fp32)
        seed, offset   [>= B] int64: each request's Philox seed and offset-window base
        slot_base, slot_span
                       position ``t`` draws at ``offset + (slot_base + t) * slot_span``
        draft_probs, slots
                       optional: ``[slots, >= K, vocab]`` fp32 rows (unit vocab stride) and
                       ``[>= B]`` int64 slot ids; position ``t``'s distribution is written to
                       ``draft_probs[slots[b], t, candidate_ids[b, t]]``. The caller zeroes
                       those rows first, as the built-in does.

    Returns ``(tokens, proposal)``: ``[B, K]`` int32 draft tokens and the
    ``[B, K, top_k]`` fp32 distributions they were drawn from (a view).
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    Metadata consumed: none. One launch: grid B x 1024 threads (512 above 128
    requests), the shipped sampler block of a B-row call. Needs the head sampler
    library loaded (``head_sampler.load_library``).

    Call site: ``ShardedDraftHeadDFlashWorker._sample_dflash2_block`` under the
    ``draft_head_sampled`` fast path (``kernels/draft_head/head.py``).

    Numerics: bitwise the built-in sampled walk, by construction (the same
    single FP32 add and ``nan_to_num``, the shipped kernel body on rows,
    parameters, offsets and alignment the built-in's call has) and by test.

    Not certified: torch.compile.
    """

    def __call__(
        self,
        edges: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        slot_base: int,
        slot_span: int,
        draft_probs: Optional[torch.Tensor] = None,
        slots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, probs = torch.ops.musespark_open_sm107_head_sampler.draft_path_sample(
            edges,
            unary,
            candidate_ids,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed,
            offset,
            slot_base,
            slot_span,
            draft_probs,
            slots,
        )
        return tokens, probs.transpose(0, 1)

    def reference(
        self,
        edges: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        slot_base: int,
        slot_span: int,
        draft_probs: Optional[torch.Tensor] = None,
        slots: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The built-in sampled walk and its ``draft_probs`` scatter, without the zeroing.

        ``dflash2_sample_candidate_paths`` with ``DFlashWorker._sample_dflash2_block``'s
        ``sample_step`` (fused sampling mode), then that method's token gather and scatter.
        """
        from tensorrt_llm._torch.models.modeling_dflash import dflash2_sample_candidate_paths

        requests, positions, _ = unary.shape
        params = (temperatures[:requests], top_ks[:requests], top_ps[:requests], min_ps[:requests])

        def sample_step(step: int, scores: torch.Tensor):
            scores = scores.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            step_offset = offset[:requests] + (slot_base + step) * slot_span
            return torch.ops.trtllm.fused_sample_from_logits_with_probs(
                scores, *params, seed[:requests], step_offset
            )

        edge_scores = unary.float()[:, :, None] + edges
        chosen, proposal = dflash2_sample_candidate_paths(edge_scores, sample_step)
        tokens = candidate_ids.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        if draft_probs is not None:
            steps = torch.arange(positions, device=edges.device)
            draft_probs[slots[:requests, None, None], steps[:, None], candidate_ids] = proposal
        return tokens.to(torch.int32), proposal

    def is_valid(
        self,
        edges: torch.Tensor,
        unary: torch.Tensor,
        candidate_ids: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        seed: torch.Tensor,
        offset: torch.Tensor,
        slot_base: int,
        slot_span: int,
        draft_probs: Optional[torch.Tensor] = None,
        slots: Optional[torch.Tensor] = None,
    ) -> None:
        if unary.dim() != 3 or unary.dtype != torch.float32 or not unary.is_contiguous():
            raise ValueError("unary must be contiguous fp32 [B, K, top_k]")
        requests, positions, top_k = unary.shape
        if tuple(edges.shape) != (requests, positions, top_k, top_k):
            raise ValueError(f"edges {tuple(edges.shape)} != [B, K, top_k, top_k]")
        if edges.dtype != torch.float32 or not edges.is_contiguous():
            raise ValueError("edges must be contiguous fp32")
        if candidate_ids.shape != unary.shape or candidate_ids.dtype != torch.int64:
            raise ValueError("candidate_ids must be int64 [B, K, top_k]")
        if not candidate_ids.is_contiguous():
            raise ValueError("candidate_ids must be contiguous")
        if (draft_probs is None) != (slots is None):
            raise ValueError("draft_probs and slots go together")
        if draft_probs is not None and (
            draft_probs.dim() != 3
            or draft_probs.stride(-1) != 1
            or draft_probs.shape[1] < positions
        ):
            raise ValueError("draft_probs must be [slots, >= K, vocab] with unit vocab stride")


draft_path_sample = DraftPathSample()
