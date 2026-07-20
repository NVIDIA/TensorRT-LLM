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
    # Stage 1: raw per-beam top-k (the per-beam adjustments are constant
    # along the vocab axis, so raw ordering == adjusted ordering).
    per_beam_vals, per_beam_tokens = topk_fn(
        logprobs.view(batch_size * beam_width_in, vocab_size), beam_width_out
    )
    per_beam_vals = per_beam_vals.view(batch_size, beam_width_in, beam_width_out)
    per_beam_tokens = per_beam_tokens.view(batch_size, beam_width_in, beam_width_out)
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
    _, selected = torch.topk(keys.view(batch_size, -1), k=beam_width_out, sorted=True, dim=-1)
    sorted_logprobs = per_beam_vals.view(batch_size, -1).gather(1, selected)
    predecessor_beams = (selected // beam_width_out).to(torch.int32)
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
    if length_penalty is not None or diversity_rate is not None:
        if length_penalty is not None:
            # Candidate generated length: active beams grow by one token this
            # step, finished beams keep their frozen length (they only append
            # pads).
            gen_lengths = beam_search_args.beam_gen_lengths[
                beam_search_args.seq_slots, :beam_width_in
            ]
            cand_gen_lengths = gen_lengths + (~finished_beams_mask).to(gen_lengths.dtype)
        # Rank by the adjusted score; keep raw cum_log_probs for storage.
        sorted_logprobs, predecessor_beam, next_tokens = beam_candidate_topk(
            logprobs,
            beam_width_out=beam_width_out,
            length_penalty=length_penalty,
            cand_gen_lengths=cand_gen_lengths,
            diversity_rate=diversity_rate,
            source_beam_indices=beam_search_args.beam_idx_arange,
            topk_fn=topk_fn,
        )
    else:
        logprobs = logprobs.view(batch_size, beam_width_in * vocab_size)
        sorted_logprobs, sorted_indices = (topk_fn or _torch_topk)(logprobs, beam_width_out)
        # torch.topk over the flattened candidates yields flat indices;
        # split them into (source beam, token id).
        flat_indices = sorted_indices.to(torch.int32)
        predecessor_beam = flat_indices // vocab_size
        next_tokens = flat_indices % vocab_size

    if cand_gen_lengths is not None:
        beam_search_args.beam_gen_lengths[beam_search_args.seq_slots, :beam_width_out] = (
            torch.gather(cand_gen_lengths, dim=1, index=predecessor_beam)
        )
    beam_search_args.predecessor_beams[beam_search_args.seq_slots, :beam_width_out] = (
        predecessor_beam
    )

    max_beam_width = beam_search_args.finished_beams.size(1)
    finished_beams = beam_search_args.finished_beams[beam_search_args.seq_slots].view(-1)

    offset_predecessor_beam = predecessor_beam + beam_search_args.seq_offsets[
        : predecessor_beam.size(0)
    ].unsqueeze(1)
    finished_beams = finished_beams[offset_predecessor_beam]
    beam_search_args.finished_beams[beam_search_args.seq_slots] = finished_beams.view(
        batch_size, max_beam_width
    )

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
    return next_tokens, softmax


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
