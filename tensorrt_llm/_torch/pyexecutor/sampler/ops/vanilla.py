# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch-native sampling kernels.

Pure tensor functions that operate on logits and probabilities with no
dependency on the sampling_utils interface or other backend implementation modules.
"""

from dataclasses import dataclass
from typing import Callable, Optional, cast

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings.executor import FinishReason

BEAM_SEARCH_PAD_TOKEN = -1

TopkFn = Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]
"""Sorted top-k over the last dim of a 2D tensor: (values, k) -> (values, indices),
values descending, int64 indices — the ``torch.topk(..., sorted=True)`` contract.
Lets callers inject an accelerated kernel (e.g. flashinfer radix top-k) without
this module depending on it."""


def _torch_topk(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.topk(values, k=k, dim=-1, sorted=True)


@dataclass(kw_only=True)
class StrategyMetadata:
    """Base class for per-strategy-group metadata passed into sample()."""


@dataclass(kw_only=True)
class BeamSearchMetadata(StrategyMetadata):
    """Stateful tensors required by beam_search_sampling_batch."""

    cache_indirection: torch.Tensor
    cache_indirection_buffer: torch.Tensor
    cum_log_probs: torch.Tensor
    new_log_probs: torch.Tensor
    seq_slots: torch.Tensor
    seq_lens: torch.Tensor
    finished_beams: torch.Tensor
    predecessor_beams: torch.Tensor
    seq_offsets: torch.Tensor
    beam_idx_arange: torch.Tensor
    beam_gen_lengths: torch.Tensor
    # --- Candidate-Beams-Array (CBA) state, present only for requests using
    # exhaustive early_stopping modes (early_stopping != 1); see
    # beam_search_sampling_batch_cba.
    end_ids: Optional[torch.Tensor] = None
    """[max_num_sequences] int32, per-slot end token id (< 0: no end token)."""
    stop_past_tokens: Optional[torch.Tensor] = None
    """[max_stop_word_length, max_num_sequences, max_beam_width] int32, the
    finish handler's rolling stop-word window (FinishReasonsHandler store).
    The CBA path reorders its beam axis by the step's predecessor beams so
    the handler's stop-word matching stays correct across beam swaps. When
    None (tests without stop words), the reorder is skipped — multi-token
    stop-word matching would then be unreliable across beam swaps."""
    prompt_lens: Optional[torch.Tensor] = None
    """[max_num_sequences] int32, per-slot prompt length."""
    original_tokens: Optional[torch.Tensor] = None
    """[max_num_sequences, max_beam_width, max_seq_len] int32, uncorrected
    per-slot tokens (see BeamSearchStore.original_tokens); read together with
    ``cache_indirection`` to snapshot finished paths into the CBA."""
    cba_tokens: Optional[torch.Tensor] = None
    """[max_num_sequences, max_beam_width, max_seq_len] int32, finished-beam
    path snapshots (generated tokens, BEAM_SEARCH_PAD_TOKEN padded)."""
    cba_cum_log_probs: Optional[torch.Tensor] = None
    """[max_num_sequences, max_beam_width] float32, raw cumulative log-probs."""
    cba_normed_scores: Optional[torch.Tensor] = None
    """[max_num_sequences, max_beam_width] float32, length-normalized scores
    (-inf: empty entry)."""
    cba_lengths: Optional[torch.Tensor] = None
    """[max_num_sequences, max_beam_width] int32, generated lengths."""
    batch_dones: Optional[torch.Tensor] = None
    """[max_num_sequences] bool, per-slot beam-search termination verdict."""
    cba_caps: Optional[torch.Tensor] = None
    """[max_num_sequences] int32, per-slot CBA capacity (the request's
    maximum beam width; C++ nBM). Differs from the step's beam_width_out for
    variable-beam-width requests."""
    original_log_probs: Optional[torch.Tensor] = None
    """[max_num_sequences, max_beam_width, max_seq_len] float32, uncorrected
    per-slot sampled log-prob per step (log-prob analog of original_tokens;
    C++ logProbsTiled). Written and read by the CBA path for logprobs."""
    cba_log_probs: Optional[torch.Tensor] = None
    """[max_num_sequences, max_beam_width, max_seq_len] float32, per-token
    log-probs of the CBA path snapshots (C++ logProbsCBA analog)."""
    max_seq_len: int = 0
    """Maximum sequence length (prompt + generated), used by the
    best-attainable-score bound of the "never" early-stopping modes."""
    max_gen_len: int = 0
    """Host-known upper bound on the generated length (including this
    step's token) across the group's requests. Bounds the width of the CBA
    path-snapshot and pool-merge operations, which would otherwise run at
    the full max_seq_len width every step. 0 means unknown (full width)."""


def top_k_top_p_sampling_batch(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temperature + optional top-k / top-p filtering + multinomial sampling.

    ``top_k=None`` (or ``vocab_size``) disables top-k filtering; ``top_p=1``
    disables top-p filtering. With both disabled this is plain temperature
    sampling.
    """
    logits_dim = logits.dim()
    assert logits_dim == 2, "logits should be 2D: [batch_size, vocab_size]"
    assert temperature > 0, "non-greedy sampling requires valid temperature"
    logits = logits / max(temperature, 1e-5)
    batch_size, vocab_size = logits.size()
    if top_k is None:
        top_k = vocab_size

    assert top_k > 1, "non-greedy sampling requires valid top_k"
    need_top_k = top_k < vocab_size
    assert top_p > 0, "non-greedy sampling requires valid top_p"
    need_top_p = top_p < 1

    if need_top_k:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1).expand(batch_size, vocab_size)
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    if need_top_p:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs_sorted = torch.softmax(sorted_logits, dim=-1)
        # NB: must NOT use out=probs_sorted here — cumulative_probs is reused as
        # the renormalization denominator below, after probs_sorted is masked.
        cumulative_probs = torch.cumsum(probs_sorted, dim=-1)
        mask_to_remove = cumulative_probs >= top_p
        last_index_to_keep = torch.searchsorted(
            mask_to_remove.to(torch.int8, non_blocking=True),
            torch.ones((1,), dtype=torch.int8, device=mask_to_remove.device).expand(
                (mask_to_remove.size(0), 1)
            ),
            right=False,
            out_int32=True,
        )
        mask_to_remove.scatter_(
            1,
            last_index_to_keep,
            torch.zeros((1,), dtype=torch.bool, device=mask_to_remove.device).expand_as(
                last_index_to_keep
            ),
        )
        probs_sorted.masked_fill_(mask_to_remove, 0.0)
        probs = torch.empty_like(probs_sorted)
        probs.scatter_(1, sorted_indices, probs_sorted)
        probs /= cumulative_probs[
            torch.arange(
                cumulative_probs.size(0), dtype=torch.int32, device=cumulative_probs.device
            ),
            last_index_to_keep.squeeze(-1),
        ].unsqueeze(-1)
        del logits
    else:
        probs = torch.softmax(logits, dim=-1)

    next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return next_tokens, probs


def greedy_search_sampling_batch(
    logits: torch.Tensor,
    *,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    next_tokens = torch.argmax(logits, dim=-1)
    softmax: Optional[torch.Tensor] = None
    if return_probs:
        softmax = torch.zeros_like(logits)
        softmax.scatter_(1, next_tokens.unsqueeze(-1), 1.0)
    return next_tokens, softmax


def _update_cache_indirection_buffer(
    cache_indirection_input: torch.Tensor,
    cache_indirection_output: torch.Tensor,
    seq_slots: torch.Tensor,
) -> None:
    assert cache_indirection_input.device == cache_indirection_output.device
    cache_indirection_input.index_copy_(0, seq_slots, cache_indirection_output[seq_slots])


def _pad_next_tokens(next_tokens: torch.Tensor, store_width: int) -> torch.Tensor:
    """Pad a [batch, beam_width_out] token tensor to the store's beam width.

    The batched sampling buffers are allocated at the maximum beam width; on
    variable-beam-width steps the op produces fewer columns, and the padding
    (BEAM_SEARCH_PAD_TOKEN) is never consumed — finalization rewrites all
    beam tokens from the corrected paths.
    """
    if next_tokens.size(1) >= store_width:
        return next_tokens
    return torch.nn.functional.pad(
        next_tokens, (0, store_width - next_tokens.size(1)), value=BEAM_SEARCH_PAD_TOKEN
    )


def beam_candidate_topk(
    logprobs: torch.Tensor,
    *,
    beam_width_out: int,
    length_penalty: "torch.Tensor | float | None" = None,
    cand_gen_lengths: Optional[torch.Tensor] = None,
    diversity_rate: "torch.Tensor | float | None" = None,
    source_beam_indices: Optional[torch.Tensor] = None,
    topk_fn: Optional[TopkFn] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-stage top-k over beam-expansion candidates with per-source-beam
    ranking adjustments, ranked by::

        (cum_log_prob + diversity_rate * source_beam_index)
            / gen_length**length_penalty

    matching the C++ kernels: the diversity term reproduces the beam-slot
    top-k of ``beamSearchKernelsTemplate.h`` (score + diversityRate * beam
    index, spreading selection across source beams), and the length term
    reproduces ``applyLengthPenalty``. One known divergence: the C++ sampler
    keeps finished beams in a separate candidate pool that the diversity term
    never touches, whereas here finished beams stay frozen in their slots, so
    their (single) candidate receives the same ``rate * slot_index`` boost as
    any other — slot position thus slightly affects how hard a finished beam
    is to evict.

    Mathematically equivalent to adjusting the full [batch, bw_in, vocab]
    candidate matrix and taking a flat top-k, but avoids touching the vocab
    axis: both adjustments are constant per source beam, so they cannot
    change the ordering *within* a beam. Any global winner is therefore
    among its own beam's top-``beam_width_out`` raw candidates. Stage 1
    takes a per-beam top-k on raw scores; stage 2 adjusts only the
    ``bw_in * bw_out`` survivors and selects the global top-k.

    Args:
        logprobs: [batch, beam_width_in, vocab] raw cumulative log-probs.
        length_penalty: scalar, or per-request tensor of shape [batch].
            None/0 disables length normalization.
        cand_gen_lengths: [batch, beam_width_in] candidate generated lengths.
            Required iff ``length_penalty`` is active.
        diversity_rate: scalar, or per-request tensor of shape [batch].
            None/0 disables the diversity adjustment.
        source_beam_indices: optional cached ``arange(beam_width_in)`` on the
            logprobs device (e.g. ``BeamSearchStore.beam_idx_arange``), used
            by the diversity adjustment; computed on the fly if omitted.

    Returns:
        (sorted_logprobs, predecessor_beams, tokens): the raw (unadjusted)
        cumulative log-probs of the selected candidates, the source-beam index
        each candidate expands from, and its token id — all of shape
        [batch, beam_width_out] (indices int32), ordered by descending
        adjusted score.
    """
    batch_size, beam_width_in, vocab_size = logprobs.shape
    topk_fn = topk_fn or _torch_topk
    # Clamps are only relevant for tiny test vocabularies: stage 1 cannot
    # exceed the vocab, the global top-k cannot exceed the pooled candidates.
    stage1_k = min(beam_width_out, vocab_size)
    beam_width_out = min(beam_width_out, beam_width_in * stage1_k)
    # Stage 1: raw per-beam top-k (the per-beam adjustments are constant
    # along the vocab axis, so raw ordering == adjusted ordering).
    per_beam_vals, per_beam_tokens = topk_fn(
        logprobs.view(batch_size * beam_width_in, vocab_size), stage1_k
    )
    per_beam_vals = per_beam_vals.view(batch_size, beam_width_in, stage1_k)
    per_beam_tokens = per_beam_tokens.view(batch_size, beam_width_in, stage1_k)
    # Stage 2: adjust only the survivors and pick the global top-k.
    keys = per_beam_vals
    if diversity_rate is not None:
        rate = (
            diversity_rate.view(-1, 1, 1)
            if isinstance(diversity_rate, torch.Tensor)
            else diversity_rate
        )
        if source_beam_indices is None:
            source_beam_indices = torch.arange(
                beam_width_in, device=logprobs.device, dtype=torch.int32
            )
        keys = keys + rate * source_beam_indices[:beam_width_in].view(1, -1, 1)
    if length_penalty is not None:
        assert cand_gen_lengths is not None, (
            "cand_gen_lengths is required when length_penalty is active"
        )
        exponent = (
            length_penalty.view(-1, 1)
            if isinstance(length_penalty, torch.Tensor)
            else length_penalty
        )
        # Candidate lengths are >= 1 by construction (active beams:
        # generated + 1; finished beams froze after generating at least one
        # token), so the power is always well-defined.
        penalty_factor = cand_gen_lengths.to(logprobs.dtype).pow(exponent)
        keys = keys / penalty_factor.unsqueeze(-1)
    _, selected = topk_fn(keys.view(batch_size, -1), beam_width_out)
    sorted_logprobs = per_beam_vals.view(batch_size, -1).gather(1, selected)
    predecessor_beams = (selected // stage1_k).to(torch.int32)
    tokens = per_beam_tokens.view(batch_size, -1).gather(1, selected).to(torch.int32)
    return sorted_logprobs, predecessor_beams, tokens


def beam_search_sampling_batch(
    logits: torch.Tensor,
    *,
    beam_width_in: int,
    beam_width_out: int,
    beam_search_args: BeamSearchMetadata,
    temperature: float | None,
    length_penalty: "torch.Tensor | float | None" = None,
    diversity_rate: "torch.Tensor | float | None" = None,
    topk_fn: Optional[TopkFn] = None,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sample beam_width tokens for each request in parallel.

    ``length_penalty`` normalizes the beam-selection ranking key as
    ``cum_log_prob / gen_length**length_penalty`` (matching the C++
    ``applyLengthPenalty``), and ``diversity_rate`` adds
    ``diversity_rate * source_beam_index`` to it (matching the C++ beam-slot
    top-k); see ``beam_candidate_topk``. The stored ``cum_log_probs`` remain
    raw. Both accept a per-request tensor of shape [batch_size] or a scalar;
    None/0 disables the respective adjustment. When ``length_penalty`` is
    active, per-beam generated lengths are maintained in place in
    ``beam_search_args.beam_gen_lengths`` (alongside the other stateful
    metadata tensors this function updates).
    """
    logits_dim = logits.dim()
    assert logits_dim == 2, "logits should be 2D: [batch_size * beam_width, vocab_size]"
    batch_size, vocab_size = logits.size()
    batch_size = batch_size // beam_width_in

    logits = logits.view(batch_size, beam_width_in, vocab_size)
    if temperature is not None and temperature != 0:
        logits = logits / max(temperature, 1e-5)
    softmax: Optional[torch.Tensor] = None
    if return_probs:
        softmax = torch.softmax(logits, dim=-1)
    _update_cache_indirection_buffer(
        beam_search_args.cache_indirection_buffer,
        beam_search_args.cache_indirection,
        beam_search_args.seq_slots,
    )
    assert batch_size == beam_search_args.seq_slots.size(0)

    logprobs = torch.log_softmax(logits, dim=-1)

    finished_beams_mask = (
        beam_search_args.finished_beams[beam_search_args.seq_slots, :beam_width_in]
        != FinishReason.NOT_FINISHED.value
    )
    finished_beams_mask_expanded = finished_beams_mask.unsqueeze(-1).expand(
        -1, -1, logprobs.size(-1)
    )
    logprobs = torch.where(finished_beams_mask_expanded, float("-inf"), logprobs)
    logprobs[..., 0] = torch.where(finished_beams_mask, 0, logprobs[..., 0])

    logprobs += beam_search_args.cum_log_probs.unsqueeze(-1)[
        beam_search_args.seq_slots, :beam_width_in
    ]

    if not isinstance(length_penalty, torch.Tensor) and not length_penalty:
        length_penalty = None  # scalar 0 (or None) disables normalization
    if not isinstance(diversity_rate, torch.Tensor) and not diversity_rate:
        diversity_rate = None  # scalar 0 (or None) disables the adjustment
    cand_gen_lengths: Optional[torch.Tensor] = None
    if length_penalty is not None:
        # Candidate generated length: active beams grow by one token this
        # step, finished beams keep their frozen length (they only append
        # pads).
        gen_lengths = beam_search_args.beam_gen_lengths[beam_search_args.seq_slots, :beam_width_in]
        cand_gen_lengths = gen_lengths + (~finished_beams_mask).to(gen_lengths.dtype)
    # Rank by the (optionally adjusted) score; keep raw cum_log_probs for
    # storage. The two-stage selection is used even without adjustments: it is
    # equivalent to a flat top-k there, and faster with the radix backend
    # (more, shorter rows parallelize better).
    sorted_logprobs, predecessor_beam, next_tokens = beam_candidate_topk(
        logprobs,
        beam_width_out=beam_width_out,
        length_penalty=length_penalty,
        cand_gen_lengths=cand_gen_lengths,
        diversity_rate=diversity_rate,
        source_beam_indices=beam_search_args.beam_idx_arange,
        topk_fn=topk_fn,
    )

    if cand_gen_lengths is not None:
        beam_search_args.beam_gen_lengths[beam_search_args.seq_slots, :beam_width_out] = (
            torch.gather(cand_gen_lengths, dim=1, index=predecessor_beam)
        )
    beam_search_args.predecessor_beams[beam_search_args.seq_slots, :beam_width_out] = (
        predecessor_beam
    )
    if beam_search_args.stop_past_tokens is not None:
        # Reorder the finish handler's rolling stop-word window to follow the
        # beam swap, so multi-token stop-word matching (which appends this
        # step's tokens to the window after this op) compares against the
        # correct per-beam history.
        window = beam_search_args.stop_past_tokens[:, beam_search_args.seq_slots, :beam_width_in]
        beam_search_args.stop_past_tokens[:, beam_search_args.seq_slots, :beam_width_out] = (
            torch.gather(
                window,
                2,
                predecessor_beam.long().unsqueeze(0).expand(window.size(0), -1, -1),
            )
        )

    finished_beams = beam_search_args.finished_beams[beam_search_args.seq_slots].view(-1)

    offset_predecessor_beam = predecessor_beam + beam_search_args.seq_offsets[
        : predecessor_beam.size(0)
    ].unsqueeze(1)
    finished_beams = finished_beams[offset_predecessor_beam]
    # Write only the beam_width_out columns produced this step: with variable
    # beam width it can be smaller than the store width (the stale view() of
    # the full width crashed on such steps).
    beam_search_args.finished_beams[beam_search_args.seq_slots, :beam_width_out] = finished_beams

    cache_indirection = beam_search_args.cache_indirection[
        beam_search_args.seq_slots, :beam_width_out
    ]
    cache_indirection_buffer = beam_search_args.cache_indirection_buffer[
        beam_search_args.seq_slots, :beam_width_in
    ]
    torch.gather(
        cache_indirection_buffer,
        dim=1,
        index=predecessor_beam.unsqueeze(2).expand(-1, -1, cache_indirection.size(2)),
        out=cache_indirection,
    )

    index = beam_search_args.seq_lens.view(-1, 1, 1).expand(-1, beam_width_out, 1)
    src = (
        beam_search_args.beam_idx_arange[:beam_width_out]
        .view(1, beam_width_out, 1)
        .expand(batch_size, beam_width_out, 1)
    )
    cache_indirection.scatter_(2, index, src)

    beam_search_args.cache_indirection[beam_search_args.seq_slots, :beam_width_out] = (
        cache_indirection
    )

    ended_predecessor_mask = torch.gather(dim=1, index=predecessor_beam, input=finished_beams_mask)
    next_tokens = torch.where(ended_predecessor_mask, BEAM_SEARCH_PAD_TOKEN, next_tokens)

    old_cum_log_probs = beam_search_args.cum_log_probs[beam_search_args.seq_slots].view(-1)
    beam_search_args.new_log_probs[beam_search_args.seq_slots, :beam_width_out] = (
        sorted_logprobs[:, :beam_width_out] - old_cum_log_probs[offset_predecessor_beam]
    )
    beam_search_args.cum_log_probs[beam_search_args.seq_slots, :beam_width_out] = sorted_logprobs[
        :, :beam_width_out
    ]
    return _pad_next_tokens(next_tokens, beam_search_args.finished_beams.size(1)), softmax


def _cba_step_math(
    # candidates (from beam_candidate_topk)
    cand_cum: torch.Tensor,  # [bs, C] float32
    cand_pred: torch.Tensor,  # [bs, C] int32
    cand_tok: torch.Tensor,  # [bs, C] int32
    # per-request state (store tensors + the group's slots)
    slots: torch.Tensor,  # [bs] int64
    seq_lens: torch.Tensor,  # [bs] int32
    snap_arange: torch.Tensor,  # [S] int64, S = bounded snapshot width
    exponent: torch.Tensor,  # [bs, 1] float32, 0 == no length penalty
    cache_indirection: torch.Tensor,
    original_tokens: torch.Tensor,
    original_log_probs: torch.Tensor,
    cum_log_probs: torch.Tensor,
    finished_beams: torch.Tensor,
    end_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    cba_caps: torch.Tensor,
    cba_normed_scores: torch.Tensor,
    cba_cum_log_probs: torch.Tensor,
    cba_lengths: torch.Tensor,
    cba_tokens: torch.Tensor,
    cba_log_probs: torch.Tensor,
    stop_past_tokens: Optional[torch.Tensor],
    # static (graph-specializing) parameters
    beam_width_in: int,
    num_beams: int,
    early_stopping: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, ...]:
    """Small-tensor math of the CBA beam-search step (see
    beam_search_sampling_batch_cba), written without data-dependent shapes so
    it can be fused by torch.compile: everything between candidate selection
    and the store writebacks. Pure — all mutations happen in the caller.
    """
    batch_size, num_candidates = cand_cum.shape
    neg_inf = float("-inf")

    harvest_mask = finished_beams[slots, :beam_width_in] != FinishReason.NOT_FINISHED.value
    end_ids_b = end_ids[slots].view(-1, 1)
    caps = cba_caps[slots].view(-1, 1)
    prompts = prompt_lens[slots].view(-1, 1)
    gen_lens = (seq_lens - prompts.view(-1)).view(-1, 1)
    cand_len = gen_lens + 1

    # -inf candidates (from harvested rows) must count as slot fillers, not
    # end candidates, to preserve the >= K actives invariant even when every
    # beam was harvested at once.
    is_end = (cand_tok == end_ids_b) & torch.isfinite(cand_cum)
    cand_rank = torch.arange(num_candidates, device=cand_cum.device).view(1, -1)

    # --- Beam slots continue with the first K non-end candidates. Scatter
    # through a K+1-wide buffer instead of masked_select (data-dependent
    # shapes cannot be compiled); non-selected entries all collide on the
    # spare column and are discarded.
    active_mask = ~is_end
    active_pos = torch.cumsum(active_mask.to(torch.int32), dim=1) - 1
    scatter_idx = torch.where(active_mask & (active_pos < num_beams), active_pos, num_beams).long()

    def _take(src: torch.Tensor) -> torch.Tensor:
        buf = src.new_zeros(batch_size, num_beams + 1)
        buf.scatter_(1, scatter_idx, src)
        return buf[:, :num_beams]

    slot_pred = _take(cand_pred)
    slot_tok = _take(cand_tok)
    slot_cum = _take(cand_cum)
    step_log_probs = slot_cum - cum_log_probs[slots, :beam_width_in].gather(1, slot_pred.long())

    # --- CBA insertion: normalized scores of the new end-token candidates.
    new_normed = cand_cum / cand_len.to(cand_cum.dtype).pow(exponent)
    eligible = is_end & (cand_rank < caps)
    new_normed = new_normed.masked_fill(~eligible, neg_inf)

    # Snapshot candidate paths through the cache indirection (the work tree
    # is rewritten by later steps). Indirection entries beyond the current
    # length are uninitialized: lanes are masked below, but gather indices
    # must be clamped in-bounds first.
    step_idx = (prompts + snap_arange.view(1, -1)).clamp(max=cache_indirection.size(-1) - 1)
    step_idx_e = step_idx.unsqueeze(1).expand(-1, beam_width_in, -1)
    ind_at = torch.gather(cache_indirection[slots, :beam_width_in].long(), 2, step_idx_e)
    ind_at = ind_at.clamp_(0, beam_width_in - 1)
    tok_at = torch.gather(original_tokens[slots, :beam_width_in], 2, step_idx_e)
    lp_at = torch.gather(original_log_probs[slots, :beam_width_in], 2, step_idx_e)
    snap_len = snap_arange.size(0)
    parent_exp = cand_pred.long().unsqueeze(-1).expand(-1, -1, snap_len)
    src_beam = torch.gather(ind_at, 1, parent_exp)
    t_valid = snap_arange.view(1, 1, -1) < gen_lens.view(-1, 1, 1)
    new_paths = torch.gather(tok_at, 1, src_beam).masked_fill(~t_valid, BEAM_SEARCH_PAD_TOKEN)
    new_lp_paths = torch.gather(lp_at, 1, src_beam).masked_fill(~t_valid, 0.0)
    end_pos = gen_lens.view(-1, 1, 1).expand(-1, num_candidates, 1).clamp(max=snap_len - 1)
    new_paths = new_paths.scatter(2, end_pos, cand_tok.unsqueeze(-1).to(new_paths.dtype))
    # the terminating token's own log-prob = candidate cum - parent cum
    parent_cum = cum_log_probs[slots, :beam_width_in].gather(1, cand_pred.long())
    new_lp_paths = new_lp_paths.scatter(2, end_pos, (cand_cum - parent_cum).unsqueeze(-1))

    # Harvested beams (stop-word finishes latched after the previous step):
    # their recorded tokens already include the terminating stop word, so the
    # snapshot is the beam's own path (parent = itself) at the current length.
    harvest_paths = torch.gather(tok_at, 1, ind_at).masked_fill(~t_valid, BEAM_SEARCH_PAD_TOKEN)
    harvest_lp_paths = torch.gather(lp_at, 1, ind_at).masked_fill(~t_valid, 0.0)
    harvest_cum = cum_log_probs[slots, :beam_width_in]
    harvest_normed = harvest_cum / gen_lens.to(harvest_cum.dtype).pow(exponent)
    harvest_normed = harvest_normed.masked_fill(~harvest_mask, neg_inf)

    # --- Pool merge: keep the best pool_width by normalized score, then
    # enforce the per-request capacity (C++ nBM replace-min equivalent).
    pool_width = cba_normed_scores.size(1)
    all_normed = torch.cat([cba_normed_scores[slots], new_normed, harvest_normed], dim=1)
    top_normed, top_i = torch.topk(all_normed, k=pool_width, sorted=True, dim=-1)
    top_normed = top_normed.masked_fill(
        torch.arange(pool_width, device=cand_cum.device).view(1, -1) >= caps, neg_inf
    )
    all_cum = torch.cat([cba_cum_log_probs[slots], cand_cum, harvest_cum], dim=1)
    all_len = torch.cat(
        [
            cba_lengths[slots],
            cand_len.expand(-1, num_candidates),
            gen_lens.expand(-1, beam_width_in),
        ],
        dim=1,
    )
    all_tokens = torch.cat([cba_tokens[slots, :, :snap_len], new_paths, harvest_paths], dim=1)
    all_lps = torch.cat([cba_log_probs[slots, :, :snap_len], new_lp_paths, harvest_lp_paths], dim=1)
    merged_cum = all_cum.gather(1, top_i)
    merged_len = all_len.gather(1, top_i)
    top_i_wide = top_i.unsqueeze(-1).expand(-1, -1, snap_len)
    merged_tokens = all_tokens.gather(1, top_i_wide)
    merged_lps = all_lps.gather(1, top_i_wide)

    # --- Done verdict (C++ batchDones): CBA full, and the best candidate's
    # attainable normalized score cannot beat the worst kept entry.
    # early_stopping != 0 (HF "never") bounds attainability with the maximum
    # length for positive penalties; otherwise the current length is the
    # correct bound (longer sequences only get less attractive).
    min_kept = top_normed.gather(1, (caps - 1).long()).view(-1)
    if early_stopping != 0:
        max_gen = (max_seq_len - prompts.view(-1)).to(cand_len.dtype)
        bound_len = torch.where(exponent.view(-1) > 0, max_gen, cand_len.view(-1))
    else:
        bound_len = cand_len.view(-1)
    best_attainable = cand_cum[:, 0] / bound_len.to(cand_cum.dtype).pow(exponent.view(-1))
    done = (min_kept > neg_inf) & (min_kept >= best_attainable)

    # Reorder the finish handler's rolling stop-word window to follow the
    # beam swap (matching stays correct across swaps).
    reordered_window = None
    if stop_past_tokens is not None:
        window = stop_past_tokens[:, slots, :beam_width_in]
        reordered_window = torch.gather(
            window, 2, slot_pred.long().unsqueeze(0).expand(window.size(0), -1, -1)
        )

    return (
        slot_pred,
        slot_tok,
        slot_cum,
        step_log_probs,
        top_normed,
        merged_cum,
        merged_len,
        merged_tokens,
        merged_lps,
        done,
        reordered_window,
    )


# Compiled lazily on first use; the first CBA-mode request of a process pays
# the inductor compile once (subsequent shapes are covered by the dynamic
# batch/snapshot-length dims marked at the call site).
_cba_step_compiled = torch.compile(_cba_step_math, dynamic=None, fullgraph=True)


def beam_search_sampling_batch_cba(
    logits: torch.Tensor,
    *,
    beam_width_in: int,
    beam_width_out: int,
    beam_search_args: BeamSearchMetadata,
    temperature: float | None,
    early_stopping: int,
    length_penalty: "torch.Tensor | float | None" = None,
    diversity_rate: "torch.Tensor | float | None" = None,
    topk_fn: Optional[TopkFn] = None,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Beam-search step with a candidate-beams array (CBA) for the exhaustive
    early_stopping modes (``early_stopping != 1``), mirroring the C++
    ``beamSearchKernels`` workflow:

    - The top ``2 * beam_width`` expansion candidates (ranked by raw
      cumulative log-prob, plus the optional diversity adjustment — the
      length penalty does NOT enter candidate ranking, matching the C++
      kernel) are split by end-token: end-token candidates ranked within the
      top ``beam_width`` are inserted into the CBA, which keeps the best
      ``beam_width`` finished paths seen so far by length-normalized score
      (path snapshots are taken eagerly since the work tree is rewritten by
      later steps). All beam slots then continue with the best non-end
      candidates, so exploration never narrows.
    - Stop words (any length) follow the C++ stop-criteria split: the finish
      handler detects them after the step and latches a per-beam finish
      reason; at the START of the next step this op harvests the latched
      beams into the CBA (their paths are complete, including the stop word)
      and masks their rows so the freed slots refill with active candidates —
      the torch equivalent of the C++ kernel coercing a finished beam into a
      top-ranked end-token candidate.
    - A request is done when the CBA is full and the best active candidate's
      attainable normalized score cannot beat the worst CBA entry.
      ``early_stopping == 0`` bounds attainability with the current length;
      any other value bounds it with ``max_seq_len`` when
      ``length_penalty > 0`` (HF's "never"), following the C++ kernel. The
      verdict is published by marking every beam slot finished, which drives
      the regular stop machinery.

    Requires the CBA fields of ``BeamSearchMetadata`` to be set.
    """
    args = beam_search_args
    assert (
        args.end_ids is not None
        and args.prompt_lens is not None
        and args.original_tokens is not None
        and args.cba_tokens is not None
        and args.cba_cum_log_probs is not None
        and args.cba_normed_scores is not None
        and args.cba_lengths is not None
        and args.batch_dones is not None
        and args.original_log_probs is not None
        and args.cba_log_probs is not None
        and args.cba_caps is not None
    ), "CBA metadata is required for early_stopping != 1"
    assert logits.dim() == 2, "logits should be 2D: [batch_size * beam_width, vocab_size]"
    batch_size, vocab_size = logits.size()
    batch_size = batch_size // beam_width_in
    num_beams = beam_width_out
    device = logits.device
    slots = args.seq_slots

    logits = logits.view(batch_size, beam_width_in, vocab_size)
    if temperature is not None and temperature != 0:
        logits = logits / max(temperature, 1e-5)
    softmax: Optional[torch.Tensor] = None
    if return_probs:
        softmax = torch.softmax(logits, dim=-1)
    _update_cache_indirection_buffer(args.cache_indirection_buffer, args.cache_indirection, slots)
    assert batch_size == slots.size(0)

    logprobs = torch.log_softmax(logits, dim=-1)
    # Beams latched finished by the finish handler after the previous step
    # are harvested into the CBA below; mask their rows so the freed slots
    # refill with active candidates from the other beams. Only STOP_WORDS
    # latches can reach this point: LENGTH fires on all beams at once (their
    # lengths are uniform here) and the END_ID flood below is all-beams too —
    # either way the request stops that same step, so there is no next step
    # to harvest them (finalize handles those paths instead).
    harvest_mask = args.finished_beams[slots, :beam_width_in] != FinishReason.NOT_FINISHED.value
    logprobs = logprobs.masked_fill(harvest_mask.unsqueeze(-1), float("-inf"))
    logprobs += args.cum_log_probs.unsqueeze(-1)[slots, :beam_width_in]

    # --- Top 2K candidates by raw score (+ diversity), two-stage. Each
    # source beam contributes at most one end-token candidate (single end
    # id), so the 2K candidates always contain at least K non-end ones for
    # the beam slots (harvested beams' rows are all -inf and count as
    # active fillers via the isfinite guard below).
    topk_fn = topk_fn or _torch_topk
    if not isinstance(diversity_rate, torch.Tensor) and not diversity_rate:
        diversity_rate = None
    # Each source beam contributes at most one finite end-token candidate, so
    # beam_width_in extra candidates on top of the slots to fill always leave
    # >= num_beams active ones — also on narrowing variable-beam-width steps
    # (beam_width_in > num_beams), where 2 * num_beams would not.
    cand_cum, cand_pred, cand_tok = beam_candidate_topk(
        logprobs,
        beam_width_out=num_beams + max(num_beams, beam_width_in),
        diversity_rate=diversity_rate,
        source_beam_indices=args.beam_idx_arange,
        topk_fn=topk_fn,
    )

    # Length penalty normalized to a per-request exponent tensor (0 == off),
    # so the compiled step math is branch-free.
    if isinstance(length_penalty, torch.Tensor):
        exponent = length_penalty.view(-1, 1).to(torch.float32)
    else:
        exponent = torch.full(
            (batch_size, 1), float(length_penalty or 0.0), dtype=torch.float32, device=device
        )

    snap_len = args.cba_tokens.size(-1)
    if args.max_gen_len > 0:
        # Snapshots and pool merges only ever touch generated positions, and
        # pool entry lengths are bounded by the running maximum generation
        # length; columns beyond it keep their (unread) previous content.
        snap_len = min(snap_len, args.max_gen_len)
    snap_arange = torch.arange(snap_len, device=device)

    # CPU callers (unit tests) take the eager function: inductor's CPU
    # compile latency would dominate, and the fusion only pays off on CUDA.
    step_fn = _cba_step_compiled if logits.is_cuda else _cba_step_math
    if logits.is_cuda:
        for t in (cand_cum, cand_pred, cand_tok, slots, args.seq_lens, exponent):
            torch._dynamo.mark_dynamic(t, 0)
        torch._dynamo.mark_dynamic(snap_arange, 0)
    (
        slot_pred,
        slot_tok,
        slot_cum,
        step_log_probs,
        top_normed,
        merged_cum,
        merged_len,
        merged_tokens,
        merged_lps,
        done,
        reordered_window,
    ) = step_fn(
        cand_cum,
        cand_pred,
        cand_tok,
        slots,
        args.seq_lens,
        snap_arange,
        exponent,
        args.cache_indirection,
        args.original_tokens,
        args.original_log_probs,
        args.cum_log_probs,
        args.finished_beams,
        args.end_ids,
        args.prompt_lens,
        args.cba_caps,
        args.cba_normed_scores,
        args.cba_cum_log_probs,
        args.cba_lengths,
        args.cba_tokens,
        args.cba_log_probs,
        args.stop_past_tokens,
        beam_width_in,
        num_beams,
        early_stopping,
        args.max_seq_len,
    )
    # --- Writebacks (kept eager: mixed advanced/basic indexing on the store
    # tensors, and the compiled math stays pure).
    args.cba_normed_scores[slots] = top_normed
    args.cba_cum_log_probs[slots] = merged_cum
    args.cba_lengths[slots] = merged_len
    args.cba_tokens[slots, :, :snap_len] = merged_tokens
    args.cba_log_probs[slots, :, :snap_len] = merged_lps
    args.batch_dones[slots] = done
    # NOT_FINISHED == 0, so the verdict maps directly onto the finish reason.
    # Flood the full row: the stop criterion reads the first py_beam_width
    # (== capacity) entries, which can exceed this step's beam_width_out for
    # variable-beam-width requests.
    args.finished_beams[slots] = (
        done.view(-1, 1).to(torch.int32) * FinishReason.END_ID.value
    ).expand(-1, args.finished_beams.size(1))
    stop_window = args.stop_past_tokens
    if reordered_window is not None and stop_window is not None:
        stop_window[:, slots, :num_beams] = reordered_window

    # --- Beam-slot state updates (same contract as beam_search_sampling_batch).
    args.predecessor_beams[slots, :num_beams] = slot_pred
    cache_indirection = args.cache_indirection[slots, :num_beams]
    cache_indirection_buffer = args.cache_indirection_buffer[slots, :beam_width_in]
    torch.gather(
        cache_indirection_buffer,
        dim=1,
        index=slot_pred.long().unsqueeze(2).expand(-1, -1, cache_indirection.size(2)),
        out=cache_indirection,
    )
    index = args.seq_lens.view(-1, 1, 1).expand(-1, num_beams, 1)
    src = args.beam_idx_arange[:num_beams].view(1, num_beams, 1).expand(batch_size, num_beams, 1)
    cache_indirection.scatter_(2, index, src)
    args.cache_indirection[slots, :num_beams] = cache_indirection

    args.new_log_probs[slots, :num_beams] = step_log_probs
    # Record this step's per-slot log-prob at the emission position so path
    # snapshots can recover per-token log-probs (analog of original_tokens).
    olp = args.original_log_probs[slots, :num_beams]
    olp.scatter_(
        2,
        args.seq_lens.view(-1, 1, 1).expand(-1, num_beams, 1).long(),
        step_log_probs.unsqueeze(-1),
    )
    args.original_log_probs[slots, :num_beams] = olp
    args.cum_log_probs[slots, :num_beams] = slot_cum
    return _pad_next_tokens(slot_tok, args.finished_beams.size(1)), softmax


def get_rejected_indices(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    generator: torch.Generator,
    draft_tokens: list[int],
) -> torch.Tensor:
    num_draft_tokens = draft_probs.size(0)
    draft_tokens = draft_tokens[:num_draft_tokens]
    token_idx = torch.arange(num_draft_tokens, dtype=torch.int32, device=generator.device)
    draft_tokens_cuda = torch.tensor(
        draft_tokens, dtype=torch.int32, pin_memory=prefer_pinned()
    ).to(device=generator.device, non_blocking=True)
    p = draft_probs[token_idx, draft_tokens_cuda]
    q = target_probs.squeeze(0)[token_idx, draft_tokens_cuda]
    accept_probs = torch.minimum(torch.ones((), device=generator.device, dtype=q.dtype), q / p)
    rejected_indices = (
        torch.rand(accept_probs.shape, generator=generator, device=accept_probs.device)
        > accept_probs
    ).nonzero()
    return rejected_indices


def sample_rejected(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    generator: torch.Generator,
    num_accepted: int,
) -> int:
    last_draft = draft_probs[num_accepted]
    last_target = target_probs[num_accepted]
    new = last_target - last_draft
    new = torch.where(new > 0, new, 0.0)
    new_token = torch.multinomial(new, num_samples=1, generator=generator).squeeze(-1)
    return cast(int, new_token.item())


# Rows whose temperature is at or below this threshold are treated as greedy.
# Contract with the spec-decoding metadata layer: greedy requests are
# normalized to a sentinel temperature strictly below this threshold (see
# DISABLE_TEMP_VAL in speculative/interface.py, which derives from it).
GREEDY_TEMPERATURE_THRESHOLD = 1e-4


def safely_apply_temperature_inplace(
    logits_inout: torch.Tensor, temp: torch.Tensor
) -> torch.Tensor:
    """Divide logits by per-row temperature in place, guarding the greedy sentinel.

    Greedy requests carry a temperature of 0 / <= ``GREEDY_TEMPERATURE_THRESHOLD``.
    Dividing by it would blow logits up to inf/nan and corrupt downstream sampling
    (argmax / softmax / multinomial). Those rows are clamped to a temperature of 1.0
    so the division is numerically safe; callers are expected to overwrite the greedy
    rows with their argmax result afterwards (e.g. via ``torch.where(is_greedy, ...)``),
    so the value used for the clamped rows here does not affect the final output.

    ``logits_inout`` is modified in place (``div_``) and also returned for
    convenience; ``temp`` is left untouched.
    """
    safe_temp = torch.where(temp <= GREEDY_TEMPERATURE_THRESHOLD, torch.ones_like(temp), temp)
    return logits_inout.div_(safe_temp.unsqueeze(dim=1))


class Fusions:
    @staticmethod
    @torch.compile(dynamic=None, fullgraph=True)
    def _gather_scatter_impl(
        dst_cuda: torch.Tensor,
        dst_index_cuda: torch.Tensor,
        src_cuda: torch.Tensor,
        src_index_cuda: torch.Tensor,
    ) -> None:
        dst_cuda[dst_index_cuda] = src_cuda[src_index_cuda]

    @staticmethod
    def gather_scatter(
        dst_cuda: torch.Tensor,
        dst_index_cuda: torch.Tensor,
        src_cuda: torch.Tensor,
        src_index_cuda: torch.Tensor,
    ) -> None:
        torch._dynamo.mark_dynamic(dst_cuda, 0)
        torch._dynamo.mark_dynamic(dst_index_cuda, 0)
        torch._dynamo.mark_dynamic(src_cuda, 0)
        torch._dynamo.mark_dynamic(src_index_cuda, 0)
        Fusions._gather_scatter_impl(dst_cuda, dst_index_cuda, src_cuda, src_index_cuda)

    @staticmethod
    @torch.compile(dynamic=None, fullgraph=True)
    def _determine_sampled_rank_impl(
        group_logprobs_cuda: torch.Tensor, sampled_logprobs_cuda: torch.Tensor
    ) -> torch.Tensor:
        sampled_rank_cuda = (
            group_logprobs_cuda.greater(sampled_logprobs_cuda).count_nonzero(dim=-1).to(torch.int32)
        )
        return sampled_rank_cuda

    @staticmethod
    def determine_sampled_rank(
        group_logprobs_cuda: torch.Tensor, sampled_logprobs_cuda: torch.Tensor
    ) -> torch.Tensor:
        torch._dynamo.mark_dynamic(group_logprobs_cuda, 0)
        torch._dynamo.mark_dynamic(sampled_logprobs_cuda, 0)
        return Fusions._determine_sampled_rank_impl(group_logprobs_cuda, sampled_logprobs_cuda)

    @staticmethod
    @torch.compile(
        dynamic=None,
        fullgraph=True,
        options=dict(
            online_softmax=True,
            split_reductions=False,
        ),
    )
    def _gather_log_softmax_impl(
        inputs_cuda: torch.Tensor, indices_cuda: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.log_softmax(
            inputs_cuda[indices_cuda],
            dim=-1,
        )

    @staticmethod
    def gather_log_softmax(inputs_cuda: torch.Tensor, indices_cuda: torch.Tensor) -> torch.Tensor:
        torch._dynamo.mark_dynamic(inputs_cuda, 0)
        torch._dynamo.mark_dynamic(indices_cuda, 0)
        return Fusions._gather_log_softmax_impl(inputs_cuda, indices_cuda)

    # --- Top-P Decay ops ---------------------------------------------------
    # Host-launch-bound per-step ops (a few dozen elements per row), fused with
    # Inductor to keep the launch count low. mode="max-autotune-no-cudagraphs":
    # cudagraphs is unsafe here (the update mutates persistent per-slot state
    # in place and the gather's output is consumed outside the compiled region;
    # cudagraph static output buffers get overwritten by subsequent replays).
    # mark_dynamic on the batch-varying dims avoids recompilation as the batch
    # composition changes. Compilation is lazy: the first decay-active request
    # pays it (roughly a second); non-decay workloads never trigger it. See
    # TorchSampler.TopPDecayStore for the feature-level semantics.

    @staticmethod
    @torch.compile(mode="max-autotune-no-cudagraphs")
    def _top_p_decay_update_impl(
        runtime_top_p: torch.Tensor,
        initial_top_p: torch.Tensor,
        top_p_decay: torch.Tensor,
        top_p_min: torch.Tensor,
        reset_ids: torch.Tensor,
        is_decay_slot: torch.Tensor,
        step_tokens: torch.Tensor,
        sampled_slots: torch.Tensor,
    ) -> None:
        active = is_decay_slot[sampled_slots]
        current = runtime_top_p[sampled_slots]
        updated = torch.where(
            step_tokens[sampled_slots] == reset_ids[sampled_slots],
            initial_top_p[sampled_slots],
            torch.maximum(current * top_p_decay[sampled_slots], top_p_min[sampled_slots]),
        )
        runtime_top_p[sampled_slots] = torch.where(active, updated, current)

    @staticmethod
    def top_p_decay_update(
        *,
        runtime_top_p: torch.Tensor,
        initial_top_p: torch.Tensor,
        top_p_decay: torch.Tensor,
        top_p_min: torch.Tensor,
        reset_ids: torch.Tensor,
        is_decay_slot: torch.Tensor,
        step_tokens: torch.Tensor,
        sampled_slots: torch.Tensor,
    ) -> None:
        """Fused in-place update of ``runtime_top_p`` for the sampled decay slots.

        Applies the Top-P Decay recurrence (see ``TorchSampler.TopPDecayStore``
        for the feature-level semantics) to every sampled row whose slot is
        decay-active per ``is_decay_slot``.

        All per-slot tensors are 1-D of length ``max_num_sequences``;
        ``step_tokens`` is a slot-indexed 1-D (possibly strided) int32 view of
        the new-tokens buffer for a fixed step/beam
        (``new_tokens[step, :, beam]``); ``sampled_slots`` is 1-D of length
        ``num_sampled`` (this iteration's rows). ``runtime_top_p`` is mutated
        in place; nothing is returned.
        """
        torch._dynamo.mark_dynamic(sampled_slots, 0)
        Fusions._top_p_decay_update_impl(
            runtime_top_p,
            initial_top_p,
            top_p_decay,
            top_p_min,
            reset_ids,
            is_decay_slot,
            step_tokens,
            sampled_slots,
        )

    @staticmethod
    @torch.compile(mode="max-autotune-no-cudagraphs")
    def _top_p_decay_gather_impl(
        runtime_top_p: torch.Tensor,
        is_decay_slot: torch.Tensor,
        static_top_p: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(is_decay_slot[slots], runtime_top_p[slots], static_top_p)

    @staticmethod
    def top_p_decay_gather(
        *,
        runtime_top_p: torch.Tensor,
        is_decay_slot: torch.Tensor,
        static_top_p: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Fused pre-sample per-row top-p gather for decay-active rows.

        Returns a new per-row tensor::

            row_top_p[i] = runtime_top_p[slots[i]]  if is_decay_slot[slots[i]]
                         = static_top_p[i]          otherwise

        ``runtime_top_p`` / ``is_decay_slot`` are per-slot arrays;
        ``static_top_p`` and ``slots`` are per-row (length = the group's
        per-step row count).
        """
        torch._dynamo.mark_dynamic(slots, 0)
        torch._dynamo.mark_dynamic(static_top_p, 0)
        return Fusions._top_p_decay_gather_impl(runtime_top_p, is_decay_slot, static_top_p, slots)
