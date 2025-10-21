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
from typing import Generic, Literal, Optional, TypeAlias, TypeVar, cast

import torch

from tensorrt_llm.sampling_params import SamplingParams

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


TemperatureOnly = tuple[Literal["temperature"], float]
TopK = tuple[Literal["top_k"], int, float]
TopP = tuple[Literal["top_p"], float, float]
TopKTopP = tuple[Literal["top_k_top_p"], int, float, float]
Greedy = tuple[Literal["greedy"], None]
GREEDY: Greedy = ("greedy", None)
Strategy = TopK | TopP | Greedy | TopKTopP | TemperatureOnly


@dataclass(frozen=True, kw_only=True)
class UtilsSamplingParams:
    """Subset of tensorrt_llm::runtime::SamplingConfig supported by sampling_utils."""

    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]


def resolve_sampling_strategy(params: UtilsSamplingParams, *, vocab_size: int) -> Strategy:
    # The semantics are specified in the doc-string of SamplingParams

    temperature = params.temperature
    top_p = params.top_p
    top_k = params.top_k

    if SamplingParams.params_imply_greedy_decoding(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    ):
        return GREEDY

    # --- resolving default values
    # NB: not greedy, hence temperature != 0 if specified
    temperature = temperature or 1.0

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
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # get the location of top_p
        # NB: Currently selecting the smallest index with cumulative_probs > top_p.
        #     Thus, top_p -> 0 resembles greedy; agreement requires torch.sort(..., stable=True).
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        # set the logits to -inf for token indices outside top_p
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float("-inf"))

    # compute probability distribution
    softmax = torch.softmax(logits, dim=-1)

    # sample from the distribution and generate result of [batch_size, 1]
    next_tokens = torch.multinomial(softmax, num_samples=1, generator=generator).squeeze(-1)
    return next_tokens, softmax


def greedy_search_sampling_batch(
    logits,
    *,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    next_tokens = torch.argmax(logits, dim=-1)
    softmax: Optional[torch.Tensor] = None
    if return_probs:
        softmax = torch.softmax(logits, dim=-1)
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
    generator: Optional[torch.Generator] = None,
    return_probs: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    match strategy:
        case ("top_k", top_k, temperature):
            tokens, softmax = top_k_sampling_batch(
                logits, top_k=top_k, temperature=temperature, generator=generator
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
    return tokens, softmax


GenericStrategyKeyType = TypeVar("GenericStrategyKeyType")


class GroupedStrategySampler(Generic[GenericStrategyKeyType], abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def strategy_grouping_key(strategy: Strategy) -> GenericStrategyKeyType:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def sample_grouped_strategies(
        group_key: GenericStrategyKeyType,
        strategies: list[Strategy],
        logits: torch.Tensor,
        *,
        group_logit_indices: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_probs: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError


class SimpleGroupedStrategySampler(GroupedStrategySampler[Strategy]):
    STRATEGY_KEY_TYPE: TypeAlias = Strategy

    @override
    @staticmethod
    def strategy_grouping_key(strategy: Strategy) -> STRATEGY_KEY_TYPE:
        return strategy

    @override
    @staticmethod
    def sample_grouped_strategies(
        group_key: STRATEGY_KEY_TYPE,
        strategies: list[Strategy],
        logits: torch.Tensor,
        *,
        group_logit_indices: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_probs: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if group_logit_indices is None:
            assert logits.size(0) == len(strategies)
        else:
            logits = logits[group_logit_indices]

        assert all(strategy == group_key for strategy in strategies), "group must be consistent"

        return sample(
            group_key,
            logits,
            generator=generator,
            return_probs=return_probs,
        )


# Inspired by https://github.com/pytorch/pytorch/issues/80577; note also the
# suggestion to consider torch.nested.
def torch_multi_arange(
    ends: torch.Tensor,
    *,
    starts: Optional[torch.Tensor] = None,
    steps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Efficiently compute torch.cat([torch.arange(b, e, d) for b, e, d in zip(starts, ends, steps)]).

    Starts, ends, steps need to share dtype and shape. Invalid ranges like range(1, 2, -1) are
    silently discarded. 'steps' defaults to 1 and 'starts' defaults to 0.
    """
    if steps is not None:
        assert ends.dtype == steps.dtype
        assert ends.shape == steps.shape
    if starts is not None:
        assert ends.dtype == starts.dtype
        assert ends.shape == starts.shape

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
        repeats = (repeats + steps - 1).div(steps, rounding_mode="floor")
    repeats = repeats.clip(0)  # ignore invalid ranges
    range_ends = repeats - 1  # last element in each range
    if steps is not None:
        range_ends *= steps
    if starts is not None:
        range_ends += starts
    prev_range_ends = range_ends.roll(1)  # last element in preceding range (or 0)
    prev_range_ends[0] = 0
    ones = (
        torch.tensor(1, dtype=ends.dtype, pin_memory=True)
        .to(device=ends.device, non_blocking=True)
        .broadcast_to(ends.shape)
    )
    if steps is None:
        steps = ones
    jumps = -prev_range_ends  # delta from one range to the next
    if starts is not None:
        jumps += starts
    seq = torch.cat((jumps.unsqueeze(-1), steps.unsqueeze(-1)), dim=1).view(-1)
    #
    # 2. Construct output via torch.repeat_interleave() and torch.cumsum()
    seq_repeats = torch.cat((ones.unsqueeze(-1), (repeats - 1).unsqueeze(-1)), dim=1).view(-1)
    seq = seq.repeat_interleave(seq_repeats)
    seq = seq.cumsum(0)
    return seq
