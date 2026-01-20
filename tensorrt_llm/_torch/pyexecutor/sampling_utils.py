# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Helper functions for sampling.

Code in this module should operate on logits and probs, without
referring to types like LlmRequest.
"""

import abc
import sys
from dataclasses import dataclass
from typing import Generic, Literal, Optional, Type, TypeAlias, TypeVar, cast

import torch

from tensorrt_llm.bindings.executor import FinishReason
from tensorrt_llm.sampling_params import SamplingParams

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


TemperatureOnly: TypeAlias = tuple[Literal["temperature"], float]
TopK: TypeAlias = tuple[Literal["top_k"], int, float]
TopP: TypeAlias = tuple[Literal["top_p"], float, float]
TopKTopP: TypeAlias = tuple[Literal["top_k_top_p"], int, float, float]
Greedy: TypeAlias = tuple[Literal["greedy"], None]
BeamSearch: TypeAlias = tuple[Literal["beam_search"], int, int, float]
GREEDY: Greedy = ("greedy", None)

Strategy: TypeAlias = TopK | TopP | Greedy | TopKTopP | TemperatureOnly | BeamSearch

BEAM_SEARCH_PAD_TOKEN = -1


@dataclass(kw_only=True)
class StrategyMetadata:
    pass


@dataclass(kw_only=True)
class BeamSearchMetadata(StrategyMetadata):
    cache_indirection: torch.Tensor
    cache_indirection_buffer: torch.Tensor
    cum_log_probs: torch.Tensor
    new_log_probs: torch.Tensor
    seq_slots: torch.Tensor
    seq_lens: torch.Tensor
    finished_beams: torch.Tensor
    predecessor_beams: torch.Tensor
    end_ids: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class UtilsSamplingParams:
    """Subset of tensorrt_llm::runtime::SamplingConfig supported by sampling_utils.

    Args:
        temperature: The temperature to use for sampling.
        top_p: The top-p to use for sampling.
        top_k: The top-k to use for sampling.
        use_beam_search: Whether to use beam search.
        beam_width_in: The beam_width of a request before the sampling step.
        beam_width_out: The beam_width of a request after the sampling step.
    """

    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    use_beam_search: Optional[bool]
    beam_width_in: Optional[int] = None
    beam_width_out: Optional[int] = None


def resolve_sampling_strategy(params: UtilsSamplingParams, *, vocab_size: int) -> Strategy:
    # The semantics are specified in the doc-string of SamplingParams

    use_beam_search = params.use_beam_search
    temperature = params.temperature
    top_p = params.top_p
    top_k = params.top_k

    if SamplingParams.params_imply_greedy_decoding(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_beam_search=use_beam_search,
    ):
        return GREEDY

    # --- resolving default values
    # NB: not greedy, hence temperature != 0 if specified
    temperature = temperature or 1.0

    # Beam search does not rely on top_p or top_k, so we can return the strategy here
    if use_beam_search:
        assert params.beam_width_in is not None and params.beam_width_out is not None, (
            "beam_width_in and beam_width_out must be specified for beam search"
        )
        return ("beam_search", params.beam_width_in, params.beam_width_out, temperature)

    # NB: not greedy, hence top_p != 0 if specified
    top_p = top_p or 1.0
    # NB: not greedy, hence top_k != 1 if specified
    #     (0 and vocab_size are equivalent)
    top_k = top_k or vocab_size

    assert top_k > 1, "non-greedy sampling requires valid top_k"
    need_top_k = top_k < vocab_size
    assert top_p > 0, "non-greedy sampling requires valid top_p"
    need_top_p = top_p < 1

    if need_top_p:
        if need_top_k:
            return ("top_k_top_p", top_k, top_p, temperature)
        return ("top_p", top_p, temperature)
    if need_top_k:
        return ("top_k", top_k, temperature)
    return ("temperature", temperature)


def top_k_sampling_batch(
    logits,
    *,
    top_k: int,
    temperature: float,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # NB: To be replaced by a more efficient implementation.
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
    # NB: To be replaced by a more efficient implementation.
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
    # NB: To be replaced by a more efficient implementation.
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

    # top-K: mask out logits not belonging to the top-K for each sample
    if need_top_k:
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1).expand(batch_size, vocab_size)

        # set the logits who is less than first top_k logits to -inf
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    # top-p: mask out logits outside the nucleus
    if need_top_p:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # compute cumulative probability distribution of each sample
        probs_sorted = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs_sorted, dim=-1)

        # get the location of top_p
        # NB: Currently selecting the smallest index with cumulative_probs >= top_p.
        #     Thus, top_p -> 0 resembles greedy; agreement requires torch.sort(..., stable=True).
        mask_to_remove = cumulative_probs >= top_p  # at least one 'True' per row
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

        # mask not selected probs
        probs_sorted.masked_fill_(mask_to_remove, 0.0)
        probs = torch.empty_like(probs_sorted)
        probs.scatter_(1, sorted_indices, probs_sorted)
        probs /= cumulative_probs[  # renormalize probs
            torch.arange(
                cumulative_probs.size(0), dtype=torch.int32, device=cumulative_probs.device
            ),  # needed for advanced indexing
            last_index_to_keep.squeeze(-1),
        ].unsqueeze(-1)
        del logits  # do not use, inconsistent with probs
    else:
        # compute probability distribution
        probs = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    return next_tokens, probs


def greedy_search_sampling_batch(
    logits,
    *,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    next_tokens = torch.argmax(logits, dim=-1)
    softmax: Optional[torch.Tensor] = None
    if return_probs:
        softmax = torch.zeros_like(logits)
        softmax.scatter_(1, next_tokens.unsqueeze(-1), 1.0)
    return next_tokens, softmax


def update_cache_indirection_buffer(
    cache_indirection_input: torch.Tensor,
    cache_indirection_output: torch.Tensor,
    seq_slots: torch.Tensor,
) -> None:
    assert cache_indirection_input.device == cache_indirection_output.device, (
        "cache_indirection_input and cache_indirection_output must be on the same device"
    )
    cache_indirection_input.index_copy_(0, seq_slots, cache_indirection_output[seq_slots])


def beam_search_sampling_batch(
    logits: torch.Tensor,
    *,
    beam_width_in: int,
    beam_width_out: int,
    beam_search_args: BeamSearchMetadata,
    temperature: float | None,
    generator: Optional[torch.Generator] = None,
    return_probs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample <beam_width> tokens for each request in parallel.
    """
    logits_dim = logits.dim()
    assert logits_dim == 2, "logits should be 2D: [batch_size * beam_width, vocab_size]"
    batch_size, vocab_size = logits.size()
    batch_size = batch_size // beam_width_in

    # compute probability distribution
    logits = logits.view(batch_size, beam_width_in, vocab_size)
    if temperature is not None and temperature != 0:
        logits = logits / max(temperature, 1e-5)
    softmax: Optional[torch.Tensor] = None
    if return_probs:
        softmax = torch.softmax(logits, dim=-1)
    # update the "input" cache indirection
    update_cache_indirection_buffer(
        beam_search_args.cache_indirection_buffer,
        beam_search_args.cache_indirection,
        beam_search_args.seq_slots,
    )
    assert batch_size == beam_search_args.seq_slots.size(0), (
        f"batch_size {batch_size} must be equal to seq_slots.size(0) {beam_search_args.seq_slots.size(0)}"
    )

    # get logprobs of each beam
    logprobs = torch.log_softmax(logits, dim=-1)

    # handle finished beams
    # Guarantee that finished beams will only sample their end_id token, with logprob 0.
    # This implies that a finished beam will not alter its score

    # mask showing which beams in the batch are finished: Shape: (batch_size, beam_width)
    finished_beams_mask = (
        beam_search_args.finished_beams[beam_search_args.seq_slots, :beam_width_in]
        != FinishReason.NOT_FINISHED.value
    )
    # expand the mask in the vocabulary dimension
    finished_beams_mask_expanded = finished_beams_mask.unsqueeze(-1).expand(
        -1, -1, logprobs.size(-1)
    )

    # we can now use torch.where to fill the logprobs of the finished beams with -inf asynchronously
    logprobs = torch.where(finished_beams_mask_expanded, float("-inf"), logprobs)
    # set the first token to 0 for finished beams. We will overwrite sampling with a padding token later.
    logprobs[..., 0] = torch.where(finished_beams_mask, 0, logprobs[..., 0])

    # Add the current cum_log_probs to the logprobs of each beam
    logprobs += beam_search_args.cum_log_probs.unsqueeze(-1)[
        beam_search_args.seq_slots, :beam_width_in
    ]

    # get the top <beam_width> logprobs across all beams
    logprobs = logprobs.view(batch_size, beam_width_in * vocab_size)
    sorted_logprobs, sorted_indices = torch.topk(logprobs, k=beam_width_out, sorted=True, dim=-1)

    next_tokens = sorted_indices.to(torch.int32)

    # Rework the past cache indirection
    # get the beam idx from which the tokens were sampled (optimal predecessor)
    predecessor_beam = next_tokens // vocab_size
    beam_search_args.predecessor_beams[beam_search_args.seq_slots, :beam_width_out] = (
        predecessor_beam
    )

    # update finished states of each beam
    max_beam_width = beam_search_args.finished_beams.size(1)
    finished_beams = beam_search_args.finished_beams[beam_search_args.seq_slots].view(-1)

    offset_predecessor_beam = predecessor_beam + (
        torch.arange(predecessor_beam.size(0), device=predecessor_beam.device).unsqueeze(1)
        * max_beam_width
    )
    finished_beams = finished_beams[offset_predecessor_beam]
    beam_search_args.finished_beams[beam_search_args.seq_slots] = finished_beams.view(
        batch_size, max_beam_width
    )

    # Update the cache indirection
    cache_indirection = beam_search_args.cache_indirection[
        beam_search_args.seq_slots, :beam_width_out
    ]
    cache_indirection_buffer = beam_search_args.cache_indirection_buffer[
        beam_search_args.seq_slots, :beam_width_in
    ]
    # Perform the swap of the cache indirections between beams
    torch.gather(
        cache_indirection_buffer,
        dim=1,
        index=predecessor_beam.unsqueeze(2).expand(-1, -1, cache_indirection.size(2)),
        out=cache_indirection,
    )

    # Prepare target values
    target_values = (
        torch.arange(
            beam_width_out * batch_size, device=cache_indirection.device, dtype=torch.int32
        )
        % beam_width_out
    )

    # seq lens is of shape (batch_size), we assume all beams have the same seq len
    # therefore we can use expand
    index = beam_search_args.seq_lens.view(-1, 1, 1).expand(-1, beam_width_out, 1)
    # index is of shape (batch_size, beam_width, 1)
    src = target_values.view(batch_size, beam_width_out, 1)
    # src is of shape (batch_size, beam_width, 1)
    # cache_indirection is of shape (batch_size, beam_width, max_seq_len)
    cache_indirection.scatter_(2, index, src)

    # copy the batched buffer back to the original buffer
    beam_search_args.cache_indirection[beam_search_args.seq_slots, :beam_width_out] = (
        cache_indirection
    )

    # project the next_tokens values to the vocab_size
    next_tokens = next_tokens % vocab_size
    ended_predecessor_mask = torch.gather(dim=1, index=predecessor_beam, input=finished_beams_mask)
    # set the finished beams to the pad token
    next_tokens = torch.where(ended_predecessor_mask, BEAM_SEARCH_PAD_TOKEN, next_tokens)

    # update the logprobs of the newly generated tokens
    # NB this is not needed if logprobs are not returned
    old_cum_log_probs = beam_search_args.cum_log_probs[beam_search_args.seq_slots].view(-1)
    beam_search_args.new_log_probs[beam_search_args.seq_slots, :beam_width_out] = (
        sorted_logprobs[:, :beam_width_out] - old_cum_log_probs[offset_predecessor_beam]
    )
    # update the beam scores
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
    # NB: ModelDrafter._pad_to_max_draft_tokens pads draft_tokens, but
    #     not draft_probs. Relying on shape of draft_probs here.
    num_draft_tokens = draft_probs.size(0)
    draft_tokens = draft_tokens[:num_draft_tokens]
    # NB: torch.arange is needed to enable "advanced indexing",
    #   cf. https://numpy.org/devdocs/user/basics.indexing.html#integer-array-indexing
    token_idx = torch.arange(num_draft_tokens, dtype=torch.int32, device=generator.device)
    draft_tokens_cuda = torch.tensor(draft_tokens, dtype=torch.int32, pin_memory=True).to(
        device=generator.device, non_blocking=True
    )
    p = draft_probs[token_idx, draft_tokens_cuda]
    q = target_probs.squeeze(0)[token_idx, draft_tokens_cuda]
    accept_probs = torch.minimum(torch.ones((), device=generator.device, dtype=q.dtype), q / p)
    # Use deterministic random generation for multi-GPU consistency
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


def sample(
    strategy: Strategy,
    logits: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    group_metadata: StrategyMetadata | None = None,
    return_probs: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
    match strategy:
        case ("top_k", top_k, temperature):
            tokens, softmax = top_k_sampling_batch(
                logits,
                top_k=top_k,
                temperature=temperature,
                generator=generator,
            )
        case ("top_p", top_p, temperature):
            tokens, softmax = top_p_sampling_batch(
                logits,
                top_p=top_p,
                generator=generator,
                temperature=temperature,
            )
        case ("top_k_top_p", top_k, top_p, temperature):
            tokens, softmax = top_k_top_p_sampling_batch(
                logits,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                generator=generator,
            )
        case ("temperature", temperature):
            tokens, softmax = temperature_sampling_batch(
                logits,
                temperature=temperature,
                generator=generator,
            )
        case ("greedy", None):
            tokens, softmax = greedy_search_sampling_batch(logits, return_probs=return_probs)
            temperature = None
        case ("beam_search", beam_width_in, beam_width_out, temperature):
            assert group_metadata is not None and isinstance(group_metadata, BeamSearchMetadata), (
                "BeamSearchMetadata is required for beam_search_sampling_batch"
            )
            tokens, softmax = beam_search_sampling_batch(
                logits,
                beam_width_in=beam_width_in,
                beam_width_out=beam_width_out,
                beam_search_args=group_metadata,
                temperature=temperature,
                generator=generator,
                return_probs=return_probs,
            )
    return tokens, softmax, temperature


GenericStrategyKeyType = TypeVar("GenericStrategyKeyType")


class GroupedStrategySampler(Generic[GenericStrategyKeyType], abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def strategy_grouping_key(strategy: Strategy, return_probs: bool) -> GenericStrategyKeyType:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_metadata_type_for_group(
        strategy_key: GenericStrategyKeyType,
    ) -> Type[StrategyMetadata] | None:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def sample_grouped_strategies(
        group_key: GenericStrategyKeyType,
        strategies: list[Strategy],
        logits: torch.Tensor,
        *,
        group_logit_indices: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        return_probs: bool,
        group_metadata: StrategyMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float | torch.Tensor | None]:
        raise NotImplementedError


class SimpleGroupedStrategySampler(GroupedStrategySampler[Strategy]):
    STRATEGY_KEY_TYPE: TypeAlias = Strategy

    @override
    @staticmethod
    def strategy_grouping_key(strategy: Strategy, return_probs: bool) -> STRATEGY_KEY_TYPE:
        return strategy

    @override
    @staticmethod
    def get_metadata_type_for_group(
        strategy_key: STRATEGY_KEY_TYPE,
    ) -> Type[StrategyMetadata] | None:
        match strategy_key:
            case ("beam_search", _, _, _):
                return BeamSearchMetadata
            case _:
                return None

    @override
    @staticmethod
    def sample_grouped_strategies(
        group_key: STRATEGY_KEY_TYPE,
        strategies: list[Strategy],
        logits: torch.Tensor,
        *,
        group_logit_indices: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        return_probs: bool,
        group_metadata: StrategyMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float | None]:
        if group_key[0] == "beam_search":
            beam_width_in = group_key[1]
        else:
            beam_width_in = 1
        if group_logit_indices is None:
            assert logits.size(0) == beam_width_in * len(strategies)
        else:
            logits = logits[group_logit_indices]

        assert all(strategy == group_key for strategy in strategies), "group must be consistent"

        return sample(
            group_key,
            logits,
            generator=generator,
            return_probs=return_probs,
            group_metadata=group_metadata,
        )


class _AcceptSyncCompute:
    pass


ACCEPT_SYNC_COMPUTE = _AcceptSyncCompute()


# Inspired by https://github.com/pytorch/pytorch/issues/80577; note also the
# suggestion to consider torch.nested.
def torch_multi_arange(
    ends: torch.Tensor,
    *,
    output_length: int | _AcceptSyncCompute,
    starts: Optional[torch.Tensor] = None,
    steps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Efficiently compute torch.cat([torch.arange(b, e, d) for b, e, d in zip(starts, ends, steps)]).

    Starts, ends, steps need to share dtype and shape. Invalid ranges like range(1, 2, -1) are
    silently discarded. 'steps' defaults to 1 and 'starts' defaults to 0.

    Provide 'output_length' to avoid synchronization when using device tensors or pass
    `ACCEPT_SYNC_COMPUTE` to explicitly accept the possibility of a device sync (for device tensors)
    or when tensors are known to reside on the host.
    """
    if steps is not None:
        assert ends.dtype == steps.dtype
        assert ends.shape == steps.shape
        assert ends.device == steps.device
    if starts is not None:
        assert ends.dtype == starts.dtype
        assert ends.shape == starts.shape
        assert ends.device == starts.device
    output_length_arg = None if isinstance(output_length, _AcceptSyncCompute) else output_length

    if ends.numel() == 0:
        return ends.clone()

    # This algorithm combines torch.repeat_interleaved() and torch.cumsum() to
    # construct the result.
    #
    # 1. Given N ranges (characterized by starts, ends, steps), construct a sequence
    #    of 2N numbers, in which the non-overlapping pairs of consecutive numbers
    #    correspond to the ranges. For a given range, the pair (a, b) is chosen such
    #    that upon torch.cumsum() application 'a' turns the last element of the
    #    preceding range into the start element for the current range and 'b' is
    #    simply the step size for the current range.
    #
    repeats = ends  # number of elements in each range
    if starts is not None:
        repeats = repeats.clone()
        repeats -= starts
    if steps is not None:
        repeats *= steps.sign()
        steps_abs = steps.abs()
        repeats = (repeats + steps_abs - 1).div(steps_abs, rounding_mode="floor")
    repeats = repeats.clip(min=0)  # ignore invalid ranges
    range_ends = repeats - 1  # last element in each range
    if steps is not None:
        range_ends *= steps
    if starts is not None:
        range_ends += starts
    prev_range_ends = range_ends.roll(1)  # last element in preceding range (or 0)
    prev_range_ends[0].fill_(0)
    ones = torch.ones((), dtype=ends.dtype, device=ends.device)
    zeros = torch.zeros((), dtype=ends.dtype, device=ends.device)
    if steps is None:
        steps = ones.broadcast_to(ends.shape)
    jumps = -prev_range_ends  # delta from one range to the next
    if starts is not None:
        jumps += starts
    #     NB: Apply correction for empty ranges
    jumps_corrections = torch.where(repeats == 0, jumps, zeros).cumsum(0, dtype=ends.dtype)
    jumps += jumps_corrections
    seq = torch.cat((jumps.unsqueeze(-1), steps.unsqueeze(-1)), dim=1).view(-1)
    #
    # 2. Construct output via torch.repeat_interleave() and torch.cumsum()
    #     NB: For a resulting empty range, repeats - 1 == -1. In this case, we
    #         should set repeats for delta and increment both to 0 instead.
    jump_repeats = torch.where(repeats == 0, zeros, ones)
    step_repeats = torch.where(repeats == 0, zeros, repeats - 1)
    seq_repeats = torch.cat((jump_repeats.unsqueeze(-1), step_repeats.unsqueeze(-1)), dim=1).view(
        -1
    )
    seq = seq.repeat_interleave(seq_repeats, output_size=output_length_arg)
    seq = seq.cumsum(0, dtype=ends.dtype)
    return seq
