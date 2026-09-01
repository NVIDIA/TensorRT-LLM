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
dependency on the sampler_strategy interface or other backend implementation modules.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(kw_only=True)
class StrategyMetadata:
    """Base class for per-strategy-group metadata passed into sample()."""


def min_p_renorm_probs(
    probs: torch.Tensor,
    min_p: torch.Tensor | float,
) -> torch.Tensor:
    """Keep tokens with prob >= ``min_p`` times the per-row max, then renormalize.

    ``min_p`` is a scalar or a per-request tensor.
    """
    max_probs = probs.max(dim=-1, keepdim=True).values
    if isinstance(min_p, torch.Tensor):
        min_p = min_p.reshape(-1, 1)
    thresholds = min_p * max_probs
    probs = probs.masked_fill(probs < thresholds, 0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def top_k_top_p_sampling_batch(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    min_p: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temperature + optional min-p / top-k / top-p filtering + multinomial sampling.

    ``top_k=None`` (or ``vocab_size``) disables top-k filtering; ``top_p=1``
    disables top-p filtering; ``min_p=0`` disables min-p filtering. With all
    disabled this is plain temperature sampling.
    """
    logits_dim = logits.dim()
    assert logits_dim == 2, "logits should be 2D: [batch_size, vocab_size]"
    assert temperature > 0, "non-greedy sampling requires valid temperature"
    logits = logits / max(temperature, 1e-5)
    batch_size, vocab_size = logits.size()
    # 0 / non-positive means "keep all" (the min_p disabled-top_k sentinel),
    # matching sanitize_top_k on the flashinfer path.
    if top_k is None or top_k <= 0:
        top_k = vocab_size

    assert top_k > 1, "non-greedy sampling requires valid top_k"
    need_top_k = top_k < vocab_size
    assert top_p > 0, "non-greedy sampling requires valid top_p"
    need_top_p = top_p < 1
    assert 0 <= min_p < 1, "non-greedy sampling requires valid min_p"
    need_min_p = min_p > 0

    if need_min_p:
        # Thresholding logits at max_logit + log(min_p) keeps the tokens with
        # prob >= min_p * max_prob; the softmax below renormalizes.
        min_values = logits.max(dim=-1, keepdim=True).values + math.log(min_p)
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

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


# Rows whose temperature is at or below this threshold are treated as greedy.
# Contract with the spec-decoding metadata layer: greedy requests are
# normalized to a sentinel temperature strictly below this threshold (see
# DISABLE_TEMP_VAL in speculative/interface.py, which derives from it).
GREEDY_TEMPERATURE_THRESHOLD = 1e-4


def occurrence_penalized_logits(
    logits: torch.Tensor,
    *,
    count: torch.Tensor,
    seen_for_repetition: torch.Tensor,
    seen_for_presence: torch.Tensor,
    repetition: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
) -> torch.Tensor:
    """The occurrence-penalty formula, shared by every backend that applies it.

    Single source of truth for the arithmetic, so the eager and CUDA-graph paths
    cannot drift apart. Callers differ only in how they gather the operands; this
    function knows nothing about slots, beams or row layouts.

    Args:
        logits: ``[..., vocab_size]`` raw logits. Not modified; the penalized values
            are returned, cast back to ``logits.dtype``.
        count: how often each token has occurred, broadcast against ``logits``.
        seen_for_repetition: mask of tokens the repetition penalty applies to. Kept
            separate from ``seen_for_presence`` because a prompt prefix excluded by
            ``prompt_ignore_length`` still counts as "seen" for repetition.
        seen_for_presence: mask of tokens presence/frequency apply to.
        repetition / presence / frequency: per-row penalty parameters, broadcast
            against ``logits``.

    Repetition is multiplicative and splits on the logit's sign -- dividing a
    negative logit would raise it -- so a value > 1 always pushes a seen token down.
    Presence and frequency are plain subtractions.
    """
    penalized = logits.float()
    repeated = torch.where(penalized < 0, penalized * repetition, penalized / repetition)
    penalized = torch.where(seen_for_repetition, repeated, penalized)
    penalized = penalized - torch.where(
        seen_for_presence,
        presence + frequency * count.to(torch.float32),
        penalized.new_zeros(()),
    )
    # Clamp only what the arithmetic above could have pushed out of range. A logit
    # that arrived non-finite -- a -inf from an embedding bias, say, since bias runs
    # before penalties -- must stay that way; clamping it to the finite minimum would
    # make a token the user masked out eligible for sampling again.
    limit = torch.finfo(logits.dtype).max
    clamped = penalized.clamp(-limit, limit)
    return torch.where(torch.isfinite(logits), clamped, logits.float()).to(logits.dtype)


class Fusions:
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
        inputs_cuda: torch.Tensor,
        indices_cuda: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        # NB: helper function for TorchSampler._process_logprobs, torch.compile is expected to avoid
        #     materializing the index select and output the results directly into the destination tensor.
        out[...] = torch.nn.functional.log_softmax(
            inputs_cuda[indices_cuda],
            dim=-1,
        )

    @staticmethod
    def gather_log_softmax_with_output(
        inputs_cuda: torch.Tensor, indices_cuda: torch.Tensor, out: torch.Tensor
    ) -> None:
        torch._dynamo.mark_dynamic(inputs_cuda, 0)
        torch._dynamo.mark_dynamic(indices_cuda, 0)
        torch._dynamo.mark_dynamic(out, 0)
        Fusions._gather_log_softmax_impl(inputs_cuda, indices_cuda, out)

    # --- Top-P Decay ops ---------------------------------------------------
    # Host-launch-bound per-step ops (a few dozen elements per row), fused with
    # Inductor to keep the launch count low. mode="max-autotune-no-cudagraphs":
    # cudagraphs is unsafe here (the update mutates persistent per-slot state
    # in place and the gather's output is consumed outside the compiled region;
    # cudagraph static output buffers get overwritten by subsequent replays).
    # mark_dynamic on the batch-varying dims avoids recompilation as the batch
    # composition changes. Compilation is lazy: the first decay-active request
    # pays it (roughly a second); non-decay workloads never trigger it. See
    # top_p_decay.TopPDecayStore for the feature-level semantics.

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

        Applies the Top-P Decay recurrence (see ``top_p_decay.TopPDecayStore``
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

    # --- Occurrence penalties (repetition / presence / frequency) -----------
    # torch/torch.compile counterpart of the C++ ``batchApplyPenalty`` kernel,
    # driven by ``PenaltyHandler`` in penalties.py, which owns the
    # workspace and documents its semantics (see ``PenaltyStore`` there).

    @staticmethod
    def update_occurrence_workspace(
        counts_cuda: torch.Tensor,
        presence_prefix_cuda: Optional[torch.Tensor],
        counted_slots: torch.Tensor,
        counted_tokens: torch.Tensor,
        prefix_slots: Optional[torch.Tensor] = None,
        prefix_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        """Scatter (slot, token) pairs into the persistent occurrence workspace.

        Args:
            counts_cuda: ``int32[num_slots, vocab_size]``, incremented in place.
            presence_prefix_cuda: ``bool[num_slots, vocab_size]`` prefix-presence
                mask, or ``None`` when no active request uses
                ``prompt_ignore_length``.
            counted_slots / counted_tokens: 1-D int64 pairs to increment in
                ``counts_cuda``.
            prefix_slots / prefix_tokens: 1-D int64 pairs to mark in
                ``presence_prefix_cuda``; ``None`` when there is nothing to mark.
        """
        if counted_slots.numel() > 0:
            ones = torch.ones(
                counted_slots.shape[0], dtype=counts_cuda.dtype, device=counts_cuda.device
            )
            # accumulate=True sums repeated (slot, token) pairs -> occurrence count.
            counts_cuda.index_put_((counted_slots, counted_tokens), ones, accumulate=True)
        if (
            presence_prefix_cuda is not None
            and prefix_slots is not None
            and prefix_tokens is not None
            and prefix_slots.numel() > 0
        ):
            # Marking a dense bool mask is idempotent, so duplicate tokens are safe.
            presence_prefix_cuda[prefix_slots, prefix_tokens] = True

    # --- Beam-search occurrence counts --------------------------------------
    # Counterpart of the per-beam workspace handling in the former C++
    # ``batchApplyPenalty`` kernel: a beam does not re-walk its history,
    # it inherits its parent beam's counts and appends the single token it just
    # emitted. ``counts_cuda`` is flat ``[num_slots * max_beam_width, vocab_size]``;
    # beam ``b`` of slot ``s`` owns row ``s * max_beam_width + b``, which collapses to
    # plain slot indexing at ``max_beam_width == 1``. A single-beam engine never calls
    # this op -- it folds inside the penalty graph instead.

    @staticmethod
    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def _update_beam_occurrence_counts_impl(
        counts_cuda: torch.Tensor,
        active_cuda: torch.Tensor,
        has_previous_token_cuda: torch.Tensor,
        beam_slot_cuda: torch.Tensor,
        new_tokens: torch.Tensor,
        predecessor_beams: torch.Tensor,
        seq_slots: torch.Tensor,
        request_num_beams: torch.Tensor,
        max_beam_width: int,
    ) -> None:
        """Re-parent every beam onto the beam it continues, then fold in its token.

        Beam ``b`` of slot ``s`` takes over the counts of ``predecessor_beams[s, b]``
        before this step's token is folded in. Slots that did not sample last step read
        the identity permutation instead, gated by ``has_previous_token``; single-beam
        slots on a beam engine are gated out by ``beam_slot_cuda``, since their
        ``predecessor_beams`` row is never written.

        ``armed`` gates per *slot* (``.unsqueeze(1)``), so beams past the current width are
        re-parented too, from a clamped and possibly stale parent -- safe because nothing
        reads such a row until the width grows to cover it, and growth re-parents it from a
        beam that was valid last step. The gate is device-side, so this runs
        unconditionally rather than pay a D2H sync.

        NB: ``fullgraph=True``, mutates ``counts_cuda`` in place, and every batch-varying
        dim-0 argument must be marked dynamic by the caller -- an unmarked peer forces the
        marked dims to specialize. Add such an argument only together with its
        ``mark_dynamic`` in ``update_beam_occurrence_counts``.
        """
        vocab = counts_cuda.size(-1)
        beam_ids = torch.arange(max_beam_width, device=counts_cuda.device)

        armed = (
            active_cuda[seq_slots] & has_previous_token_cuda[seq_slots] & beam_slot_cuda[seq_slots]
        ).unsqueeze(1)
        # Beams past the current width hold stale parents; clamping keeps the
        # gather in range. Their rows are never read while masked out, and a later
        # beam-width growth only ever re-gathers from a valid parent row.
        parent = predecessor_beams[seq_slots].to(torch.int64).clamp(0, max_beam_width - 1)
        src_beam = torch.where(armed, parent, beam_ids.expand_as(parent))
        base = seq_slots.unsqueeze(1) * max_beam_width
        counts_cuda.index_copy_(
            0,
            (base + beam_ids).reshape(-1),
            counts_cuda.index_select(0, (base + src_beam).reshape(-1)),
        )

        # Same masked flat scatter as the single-beam fold, fanned out over the
        # beam axis: masked entries add 0 at counts[row, 0], so inactive, unarmed,
        # out-of-layout and padded-token beams are no-ops. A beam the previous step
        # did not produce carries BEAM_SEARCH_PAD_TOKEN (-1), which the range check
        # rejects -- that is what confines the fold to the beams that actually
        # sampled, since ``request_num_beams`` is the (wider) row-layout width.
        previous_token = new_tokens[0, seq_slots, :].to(torch.int64)  # [R, max_beam_width]
        fold_ok = (
            (active_cuda[seq_slots] & has_previous_token_cuda[seq_slots]).unsqueeze(1)
            & (beam_ids.unsqueeze(0) < request_num_beams.unsqueeze(1))
            & (previous_token >= 0)
            & (previous_token < vocab)
        )
        rows = base + beam_ids  # [R, max_beam_width]
        flat_index = rows * vocab + torch.where(
            fold_ok, previous_token, previous_token.new_zeros(())
        )
        counts_cuda.view(-1).scatter_add_(
            0, flat_index.reshape(-1), fold_ok.reshape(-1).to(counts_cuda.dtype)
        )

    @staticmethod
    def update_beam_occurrence_counts(
        counts_cuda: torch.Tensor,
        active_cuda: torch.Tensor,
        has_previous_token_cuda: torch.Tensor,
        beam_slot_cuda: torch.Tensor,
        new_tokens: torch.Tensor,
        predecessor_beams: torch.Tensor,
        seq_slots: torch.Tensor,
        request_num_beams: torch.Tensor,
        max_beam_width: int,
    ) -> None:
        """Advance the per-beam occurrence counts by one step, in place.

        Re-parents every beam onto the beam it continues, then folds in the token
        each beam sampled last step. Must run before anything reads the counts for
        this step, and before this step's sampling overwrites ``predecessor_beams``.
        This wrapper only marks the batch-varying dims dynamic; the work is in
        ``_update_beam_occurrence_counts_impl``.

        Also covers single-beam requests sharing a beam engine: ``beam_slot_cuda``
        turns their re-parent into the identity, and ``request_num_beams == 1``
        confines their fold to beam 0. Context requests are likewise laid out at
        one beam and read as such.

        Args:
            counts_cuda: ``int32[num_slots * max_beam_width, vocab_size]`` workspace.
            active_cuda / has_previous_token_cuda: per-slot gates, length ``num_slots``.
            new_tokens: ``[max_tokens, num_slots, max_beam_width]`` device buffer
                holding the previous step's sampled token per beam.
            predecessor_beams: ``int32[num_slots, max_beam_width]``, the parent beam
                of each beam as written by the previous step's beam search.
            seq_slots: ``int64[R]`` slot per request.
            request_num_beams: ``[R]`` row-layout beam width per request, i.e. the
                static admission width the logits rows are laid out at (1 for a
                context request), not the per-iteration width. Beams at or past it
                are skipped; under a growing ``beam_width_array`` the beams between
                the per-iteration and the layout width are skipped instead by their
                BEAM_SEARCH_PAD_TOKEN.
        """
        if seq_slots.numel() == 0:
            return
        # Batch-varying dim-0 tensors; mark every one, or an unmarked peer forces the
        # marked dims to specialize (cf. apply_batched_occurrence_penalties). The other
        # arguments keep dim 0 == num_slots (fixed), so they must NOT be marked.
        torch._dynamo.mark_dynamic(seq_slots, 0)
        torch._dynamo.mark_dynamic(request_num_beams, 0)
        Fusions._update_beam_occurrence_counts_impl(
            counts_cuda,
            active_cuda,
            has_previous_token_cuda,
            beam_slot_cuda,
            new_tokens,
            predecessor_beams,
            seq_slots,
            request_num_beams,
            max_beam_width,
        )

    # fullgraph=True is safe here: 4 cache entries per process against a default limit of 8
    # -- the size-1 specialization of the dynamic dim, times the `prefix_seen_cuda` Optional
    # flipping on first use. The beam parameters are per-engine constants and cost nothing.
    @staticmethod
    @torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
    def _apply_occurrence_penalties_impl(
        logits: torch.Tensor,
        counts_cuda: torch.Tensor,
        prefix_seen_cuda: Optional[torch.Tensor],
        active_cuda: torch.Tensor,
        has_previous_token_cuda: torch.Tensor,
        new_tokens: torch.Tensor,
        seq_slots: torch.Tensor,
        request_offsets: torch.Tensor,
        request_num_steps: torch.Tensor,
        repetition_cuda: torch.Tensor,
        presence_cuda: torch.Tensor,
        frequency_cuda: torch.Tensor,
        request_num_beams: Optional[torch.Tensor],
        max_beam_width: int,
        fold_pending: bool,
    ) -> None:
        vocab = logits.size(-1)

        # Fold the device-pending sampled token into the persistent counts, once per armed
        # active slot, before the gather reads them, via one flat scatter_add. Masked entries
        # add 0 at counts[slot, 0], so inactive/unarmed/out-of-range slots are no-ops.
        # Only reached on a single-beam engine, so ``max_beam_width`` is 1 and the row
        # scaling folds away. A beam engine has to re-parent before folding, which it
        # cannot do here, so it folds in ``update_beam_occurrence_counts`` instead.
        if fold_pending:
            slot_rows = seq_slots * max_beam_width
            previous_token = new_tokens[0, seq_slots, 0].to(torch.int64)
            fold_ok = (
                active_cuda[seq_slots]
                & has_previous_token_cuda[seq_slots]
                & (request_num_steps > 0)
                & (previous_token >= 0)
                & (previous_token < vocab)
            )
            flat_index = slot_rows * vocab + torch.where(
                fold_ok, previous_token, previous_token.new_zeros(())
            )
            counts_cuda.view(-1).scatter_add_(0, flat_index, fold_ok.to(counts_cuda.dtype))

        # Map each logits row to its owning request with a broadcasted range comparison.
        # This is O(T * R), but T and R are both small (rows per step x requests) and the
        # whole thing fuses into the surrounding elementwise graph, so it measures faster
        # than either a searchsorted lookup or a repeat_interleave expansion. Notably
        # torch.repeat_interleave must NOT be used here: its output length is
        # sum(num_steps), which -- without an explicit host-provided output_size -- torch
        # reads back from the device, and that per-step D2H sync destroys the overlap
        # between the sampler's host work and the model forward (measured ~20x slower).
        rows = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)  # [T, 1]
        # A request owns num_steps * num_beams rows, laid out beam-major / step-minor.
        # request_num_beams is None on a single-beam engine, collapsing the span to
        # num_steps and the counts row to plain slot indexing.
        span = (
            request_num_steps
            if request_num_beams is None
            else request_num_steps * request_num_beams
        )
        owned = (rows >= request_offsets) & (rows < request_offsets + span)  # [T, R]
        row_owned = owned.any(dim=1)  # [T]
        row_slot = (owned * seq_slots).sum(dim=1)  # [T]; slot per row, 0 for unowned
        row_active = row_owned & active_cuda[row_slot]

        if request_num_beams is None:
            row_counts = row_slot
        else:
            # Recover the beam from the row's offset within its request: beam-major means
            # beam = local_index // num_steps.
            local = (owned * (rows - request_offsets)).sum(dim=1)  # [T]
            steps = (owned * request_num_steps).sum(dim=1).clamp(min=1)  # [T]; avoid //0
            row_counts = row_slot * max_beam_width + local // steps

        count = counts_cuda[row_counts]
        rep = repetition_cuda[row_slot].unsqueeze(1)
        pre = presence_cuda[row_slot].unsqueeze(1)
        freq = frequency_cuda[row_slot].unsqueeze(1)

        seen = count > 0
        if prefix_seen_cuda is not None:
            # Prompt-ignore-prefix tokens count for repetition only, not presence/frequency.
            seen = seen | prefix_seen_cuda[row_slot]

        penalized = occurrence_penalized_logits(
            logits,
            count=count,
            seen_for_repetition=seen,
            seen_for_presence=count > 0,
            repetition=rep,
            presence=pre,
            frequency=freq,
        )
        # Cast before the select so inactive rows stay bit-identical, then write in place.
        logits.copy_(torch.where(row_active.unsqueeze(1), penalized, logits))

    @staticmethod
    def apply_batched_occurrence_penalties(
        logits: torch.Tensor,
        counts_cuda: torch.Tensor,
        presence_prefix_cuda: Optional[torch.Tensor],
        active_cuda: torch.Tensor,
        has_previous_token_cuda: torch.Tensor,
        new_tokens: torch.Tensor,
        seq_slots: torch.Tensor,
        request_offsets: torch.Tensor,
        request_num_steps: torch.Tensor,
        repetition_cuda: torch.Tensor,
        presence_cuda: torch.Tensor,
        frequency_cuda: torch.Tensor,
        request_num_beams: Optional[torch.Tensor] = None,
        max_beam_width: int = 1,
        fold_pending: bool = True,
    ) -> None:
        """Apply occurrence penalties to ``logits`` in place, before temperature handling.

        Args:
            logits: ``[T, vocab_size]`` packed generated-token logits, where
                ``T == sum(num_steps * num_beams)``. Request ``r`` owns the rows
                ``request_offsets[r] + beam * num_steps[r] + step``, i.e. beam-major /
                step-minor; rows no request owns are left bit-identical. Modified in place.
            counts_cuda / presence_prefix_cuda: the occurrence workspace; see
                ``PenaltyHandler.PenaltyStore`` for their semantics.
            active_cuda / has_previous_token_cuda / repetition_cuda / presence_cuda /
                frequency_cuda: per-slot buffers of length ``max_num_sequences``.
            new_tokens: ``[max_tokens, max_num_sequences, max_beam_width]`` device
                buffer holding the previous step's sampled token.
            seq_slots: ``int64[R]`` slot per request.
            request_offsets / request_num_steps: ``[R]`` device tensors, already
                staged by the caller. The owned spans must not overlap, but they need
                not be ordered, and rows they skip are left bit-identical.
            request_num_beams: ``[R]`` row-layout beam width per request -- the static
                admission width ModelEngine lays the rows out at, which under a growing
                ``beam_width_array`` exceeds the per-iteration width -- so a row can be
                mapped back to the beam that owns it. None on a single-beam engine.
            max_beam_width: the engine's beam width, the stride of the counts rows.
            fold_pending: whether to fold the device-pending token here. False on a beam
                engine, where ``update_beam_occurrence_counts`` has already re-parented
                and folded every slot.

        The last three are compile-time constants to Dynamo (an Optional and two Python
        scalars), so each engine specializes to its own graph and neither carries the
        other's branches.

        All heavy lifting is fused into the single compiled ``_apply_occurrence_penalties_impl``
        graph; this wrapper only marks the batch-varying dims dynamic.
        """
        if seq_slots.numel() == 0 or logits.size(0) == 0:
            return

        # Batch-varying dim-0 tensors; mark every one, or an unmarked peer forces the marked
        # dims to specialize (ConstraintViolationError under dynamic=None). counts/active/params
        # keep dim 0 == max_num_sequences (fixed) and new_tokens dim 1 == max_num_sequences.
        torch._dynamo.mark_dynamic(logits, 0)
        torch._dynamo.mark_dynamic(seq_slots, 0)
        torch._dynamo.mark_dynamic(request_offsets, 0)
        torch._dynamo.mark_dynamic(request_num_steps, 0)
        if request_num_beams is not None:
            torch._dynamo.mark_dynamic(request_num_beams, 0)
        Fusions._apply_occurrence_penalties_impl(
            logits,
            counts_cuda,
            presence_prefix_cuda,
            active_cuda,
            has_previous_token_cuda,
            new_tokens,
            seq_slots,
            request_offsets,
            request_num_steps,
            repetition_cuda,
            presence_cuda,
            frequency_cuda,
            request_num_beams,
            max_beam_width,
            fold_pending,
        )
