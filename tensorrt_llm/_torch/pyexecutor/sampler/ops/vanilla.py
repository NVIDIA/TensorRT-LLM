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

_BEAM_SEARCH_PAD_TOKEN = -1


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


def top_k_sampling_batch(
    logits: torch.Tensor,
    *,
    top_k: int,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return top_k_top_p_sampling_batch(
        logits,
        top_k=top_k,
        temperature=temperature,
        generator=generator,
        top_p=1,
    )


def top_p_sampling_batch(
    logits: torch.Tensor,
    *,
    top_p: float,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return top_k_top_p_sampling_batch(
        logits,
        top_p=top_p,
        top_k=logits.size(1),
        temperature=temperature,
        generator=generator,
    )


def temperature_sampling_batch(
    logits: torch.Tensor,
    *,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return top_k_top_p_sampling_batch(
        logits,
        top_p=1,
        top_k=logits.size(1),
        temperature=temperature,
        generator=generator,
    )


def top_k_top_p_sampling_batch(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_dim = logits.dim()
    assert logits_dim == 2, "logits should be 2D: [batch_size, vocab_size]"
    assert temperature > 0, "non-greedy sampling requires valid temperature"
    logits = logits / max(temperature, 1e-5)
    batch_size, vocab_size = logits.size()

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
    next_tokens = torch.where(ended_predecessor_mask, _BEAM_SEARCH_PAD_TOKEN, next_tokens)

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


_GREEDY_TEMPERATURE_THRESHOLD = 1e-4


def greedy(
    logits: torch.Tensor,
    *,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Greedy decoding; returns exact one-hot when return_probs=True."""
    return greedy_search_sampling_batch(logits, return_probs=return_probs)


def _safely_apply_temperature_inplace(
    logits_inout: torch.Tensor, temp: torch.Tensor
) -> torch.Tensor:
    """Divide logits by per-row temperature in place, guarding the greedy sentinel.

    Greedy requests carry a temperature of 0 / <= ``_GREEDY_TEMPERATURE_THRESHOLD``.
    Dividing by it would blow logits up to inf/nan and corrupt downstream sampling
    (argmax / softmax / multinomial). Those rows are clamped to a temperature of 1.0
    so the division is numerically safe; callers are expected to overwrite the greedy
    rows with their argmax result afterwards (e.g. via ``torch.where(is_greedy, ...)``),
    so the value used for the clamped rows here does not affect the final output.

    ``logits_inout`` is modified in place (``div_``) and also returned for
    convenience; ``temp`` is left untouched.
    """
    safe_temp = torch.where(temp <= _GREEDY_TEMPERATURE_THRESHOLD, torch.ones_like(temp), temp)
    return logits_inout.div_(safe_temp.unsqueeze(dim=1))


def _apply_top_k_top_p(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = top_k_mask.clamp(min=0)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))
    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)


def _random_sample(probs: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_()
    return probs.div_(q).argmax(dim=-1).view(-1)


def forward_native_sampling(
    logits: torch.Tensor,
    k: Optional[torch.Tensor],
    p: Optional[torch.Tensor],
) -> torch.Tensor:
    logits = _apply_top_k_top_p(logits, k, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    return _random_sample(probs)


def compute_probs_from_logits_op(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: Optional[torch.Tensor],
    top_p: Optional[torch.Tensor],
) -> torch.Tensor:
    """Pure-PyTorch CPU fallback for probability computation."""
    is_greedy = temperatures <= _GREEDY_TEMPERATURE_THRESHOLD
    # Greedy rows must pick the argmax of the *original* logits (before temperature).
    # Capture the argmax up front; _safely_apply_temperature_inplace then guards the
    # division against the greedy sentinel.
    argmax_ids = logits.argmax(dim=-1, keepdim=True)

    logits = _safely_apply_temperature_inplace(logits, temperatures)
    logits = _apply_top_k_top_p(logits, top_k, top_p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # Turn the greedy rows into a one-hot at argmax by editing `probs` in place,
    # instead of building a full [batch, vocab] one-hot buffer and a [batch, vocab]
    # torch.where copy. The torch.where here only runs on a [batch, 1] tensor.
    # NB: argwhere/index-select on the greedy rows would give a data-dependent shape
    # that breaks the surrounding torch.compile graph, so we keep it dense.
    greedy_col = is_greedy.unsqueeze(1)
    new_at_argmax = torch.where(greedy_col, 1.0, probs.gather(1, argmax_ids))
    probs.masked_fill_(greedy_col, 0.0)
    probs.scatter_(1, argmax_ids, new_at_argmax)
    return probs


class _Fusions:
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
        _Fusions._gather_scatter_impl(dst_cuda, dst_index_cuda, src_cuda, src_index_cuda)

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
        return _Fusions._determine_sampled_rank_impl(group_logprobs_cuda, sampled_logprobs_cuda)

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
        return _Fusions._gather_log_softmax_impl(inputs_cuda, indices_cuda)
