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
from typing import Optional, cast

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings.executor import FinishReason

BEAM_SEARCH_PAD_TOKEN = -1


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


def beam_search_sampling_batch(
    logits: torch.Tensor,
    *,
    beam_width_in: int,
    beam_width_out: int,
    beam_search_args: BeamSearchMetadata,
    temperature: float | None,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sample beam_width tokens for each request in parallel."""
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

    logprobs = logprobs.view(batch_size, beam_width_in * vocab_size)
    sorted_logprobs, sorted_indices = torch.topk(logprobs, k=beam_width_out, sorted=True, dim=-1)

    next_tokens = sorted_indices.to(torch.int32)

    predecessor_beam = next_tokens // vocab_size
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

    next_tokens = next_tokens % vocab_size
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


# --- Occurrence penalties (repetition / presence / frequency) -----------------
# torch/torch.compile counterpart of the C++ ``batchApplyPenalty`` kernel, driven by
# ``_OccurrencePenaltyHandler`` in sampler.py. The per-row math lives in an
# Inductor-compiled ``_impl``; the wrapper does the ragged row expansion, the eager count
# fold, and ``mark_dynamic`` -- the same idiom as the ``Fusions`` ops above.
def update_occurrence_workspace(
    counts: torch.Tensor,
    presence_prefix: Optional[torch.Tensor],
    counted_slots: torch.Tensor,
    counted_tokens: torch.Tensor,
    prefix_slots: torch.Tensor,
    prefix_tokens: torch.Tensor,
) -> None:
    """Fold newly-committed tokens into the persistent occurrence workspace.

    Mirrors the logical accumulation ``batchApplyPenalty`` performs in its
    ``penaltyWorkspace``: tokens in the ignored prompt prefix
    ``[0, prompt_ignore_length)`` set the presence mask only (they count for
    repetition but not for presence/frequency), while all other tokens (the rest of
    the prompt plus every generated token) increment the occurrence counts.

    Arguments (all index tensors are 1-D and pre-split on the host):
      counts: ``int32[num_slots, vocab_size]`` occurrence counts, updated in place.
      presence_prefix: dense ``bool[num_slots, vocab_size]`` prefix-presence mask, or
        ``None`` when no active request uses ``prompt_ignore_length``.
      counted_slots / counted_tokens: (slot, token) pairs to increment in ``counts``.
      prefix_slots / prefix_tokens: (slot, token) pairs to mark in ``presence_prefix``.
    """
    if counted_slots.numel() > 0:
        ones = torch.ones(counted_slots.shape[0], dtype=counts.dtype, device=counts.device)
        # accumulate=True sums repeated (slot, token) pairs -> occurrence count.
        counts.index_put_((counted_slots, counted_tokens), ones, accumulate=True)
    if presence_prefix is not None and prefix_slots.numel() > 0:
        # Dense bool mask -> marking is idempotent and duplicate-safe (no bit carry,
        # unlike a packed bitmap which would need an atomic OR).
        presence_prefix[prefix_slots, prefix_tokens] = True


# fullgraph=True is safe here: served model has fixed shapes and compiles ~2 graphs,
# well under the default limit (8)
@torch.compile(fullgraph=True, mode="max-autotune-no-cudagraphs")
def _apply_occurrence_penalties_impl(
    logits: torch.Tensor,
    counts: torch.Tensor,
    prefix_seen: Optional[torch.Tensor],
    active: torch.Tensor,
    has_previous_token: torch.Tensor,
    new_tokens: torch.Tensor,
    seq_slots: torch.Tensor,
    request_offsets: torch.Tensor,
    request_num_steps: torch.Tensor,
    repetition: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
) -> None:
    """Fold the pending token and apply occurrence penalties over the full logits batch, in place.

    ``logits`` is the packed generated-token logits ``[T, vocab]`` for *all* sampling
    requests; request ``r`` (slot ``seq_slots[r]``) owns the contiguous rows
    ``[request_offsets[r], request_offsets[r] + request_num_steps[r])``. ``counts`` /
    ``prefix_seen`` / ``repetition`` / ``presence`` / ``frequency`` / ``active`` are
    per-slot (length ``max_num_sequences``). Rows whose slot is inactive are written
    back bit-identical.
    """
    vocab = logits.size(-1)

    # Fold the device-pending sampled token into the persistent counts, once per armed
    # active slot, before the gather reads them, via one flat scatter_add. Masked entries
    # add 0 at counts[slot, 0], so inactive/unarmed/out-of-range slots are no-ops.
    previous_token = new_tokens[0, seq_slots, 0].to(torch.int64)
    fold_ok = (
        active[seq_slots]
        & has_previous_token[seq_slots]
        & (request_num_steps > 0)
        & (previous_token >= 0)
        & (previous_token < vocab)
    )
    flat_index = seq_slots * vocab + torch.where(
        fold_ok, previous_token, previous_token.new_zeros(())
    )
    counts.view(-1).scatter_add_(0, flat_index, fold_ok.to(counts.dtype))

    rows = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)  # [T, 1]
    owned = (rows >= request_offsets) & (rows < request_offsets + request_num_steps)  # [T, R]
    row_owned = owned.any(dim=1)  # [T]
    row_slot = (owned * seq_slots).sum(dim=1)  # [T]; slot per row, 0 for unowned (masked below)
    row_active = row_owned & active[row_slot]

    count = counts[row_slot]
    rep = repetition[row_slot].unsqueeze(1)
    pre = presence[row_slot].unsqueeze(1)
    freq = frequency[row_slot].unsqueeze(1)

    seen = count > 0
    if prefix_seen is not None:
        # Prompt-ignore-prefix tokens count for repetition only, not presence/frequency.
        seen = seen | prefix_seen[row_slot]

    penalized = logits.float()
    repeated = torch.where(penalized < 0, penalized * rep, penalized / rep)
    penalized = torch.where(seen, repeated, penalized)
    penalized = penalized - torch.where(
        count > 0,
        pre + freq * count.to(torch.float32),
        penalized.new_zeros(()),
    )
    limit = torch.finfo(logits.dtype).max
    penalized = penalized.clamp(-limit, limit).to(logits.dtype)
    # Cast before the select so inactive rows stay bit-identical, then write in place.
    logits.copy_(torch.where(row_active.unsqueeze(1), penalized, logits))


def apply_batched_occurrence_penalties(
    logits: torch.Tensor,
    counts: torch.Tensor,
    presence_prefix: Optional[torch.Tensor],
    active: torch.Tensor,
    has_previous_token: torch.Tensor,
    new_tokens: torch.Tensor,
    seq_slots: torch.Tensor,
    request_offsets: torch.Tensor,
    request_num_steps: torch.Tensor,
    repetition: torch.Tensor,
    presence: torch.Tensor,
    frequency: torch.Tensor,
) -> None:
    """Apply occurrence penalties to ``logits`` in place, before temperature handling.

    ``logits`` is the packed generated-token logits ``[sum(num_steps * num_beams),
    vocab_size]``. Request ``r`` (slot ``seq_slots[r]``) owns the rows
    ``request_offsets[r] + step`` for ``step in [0, request_num_steps[r])``; every
    such addressed row observes the same folded occurrence history, and rows not
    addressed by any request are left bit-identical.

    All heavy lifting is fused into the single compiled ``_apply_occurrence_penalties_impl``
    graph; this wrapper only moves the small ``[R]`` request metadata to device and marks
    the batch-varying dims dynamic.
    """
    if seq_slots.numel() == 0 or logits.size(0) == 0:
        return

    device = logits.device
    req_offsets = request_offsets.to(device=device, non_blocking=True)
    req_num_steps = request_num_steps.to(device=device, non_blocking=True)

    # Batch-varying dim-0 tensors; mark every one, or an unmarked peer forces the marked
    # dims to specialize (ConstraintViolationError under dynamic=None). counts/active/params
    # keep dim 0 == max_num_sequences (fixed) and new_tokens dim 1 == max_num_sequences.
    torch._dynamo.mark_dynamic(logits, 0)
    torch._dynamo.mark_dynamic(seq_slots, 0)
    torch._dynamo.mark_dynamic(req_offsets, 0)
    torch._dynamo.mark_dynamic(req_num_steps, 0)
    _apply_occurrence_penalties_impl(
        logits,
        counts,
        presence_prefix,
        active,
        has_previous_token,
        new_tokens,
        seq_slots,
        req_offsets,
        req_num_steps,
        repetition,
        presence,
        frequency,
    )
