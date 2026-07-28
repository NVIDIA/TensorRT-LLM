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

"""Shared foundation for the sampler package.

Common code that the feature modules (beam search, penalties, token ban, ...)
and the pipeline all depend on: generic helpers, per-request queries that read
``LlmRequest`` / ``SamplingConfig``, and the ``UtilsSamplingParams`` value type.

This is the base layer: it may depend on ``ops`` and ``sampler_strategy`` but
nothing above it imports back into it, keeping the sampler package's import
graph acyclic.
"""

from typing import List, Optional, TypeVar, cast

import torch

from ..llm_request import LlmRequest
from .sampler_strategy import Strategy, UtilsSamplingParams, resolve_sampling_strategy

T = TypeVar("T")


def _unwrap_singleton(p: Optional[List[T]]) -> Optional[T]:
    if p is None:
        return None
    (t,) = p
    return t


def int_tensor(shape: tuple[int, ...], device: str = "cuda") -> torch.Tensor:
    return torch.empty(shape, dtype=torch.int, device=device)


def add_token(
    request: LlmRequest, new_tokens: list[list[list[int]]], *, beam_idx: int, step: int = 0
) -> int:
    # NB: Accessing nested lists faster than torch.Tensor or numpy.ndarray
    seq_slot = request.py_seq_slot
    assert seq_slot is not None
    new_token = new_tokens[step][seq_slot][beam_idx]
    request.add_new_token(new_token, beam_idx)
    return new_token


def _get_beam_width_in(request: LlmRequest) -> int:
    return (
        1
        if request.is_context_init_state
        else request.get_beam_width_by_iter(for_next_iteration=False)
    )


def _get_beam_width_out(request: LlmRequest) -> int:
    return request.get_beam_width_by_iter(for_next_iteration=True)


def _get_max_beam_width(request: LlmRequest) -> int:
    sampling_config = request.sampling_config
    max_beam_width = cast(int, sampling_config.beam_width)
    if sampling_config.beam_width_array is not None:
        max_beam_width = max(
            max_beam_width,
            cast(
                int, torch.tensor(sampling_config.beam_width_array, dtype=torch.int32).max().item()
            ),
        )
    return max_beam_width


def _request_get_sampling_params(request: LlmRequest) -> UtilsSamplingParams:
    sampling_config = request.sampling_config
    # These sampling fields live on the C++ SamplingConfig as optional<vector<T>>
    # (a shape designed for the batched TRT-LLM sampler); the torch sampler consumes
    # them per request, so we unwrap the singleton lists into scalars here. When the
    # TRT-LLM sampler is removed, this SamplingConfig-based plumbing should be removed
    # too in favor of reading the values directly from the per-request params.
    temperature = _unwrap_singleton(cast(Optional[list[float]], sampling_config.temperature))
    top_p = _unwrap_singleton(cast(Optional[list[float]], sampling_config.top_p))
    top_k = _unwrap_singleton(cast(Optional[list[int]], sampling_config.top_k))
    top_p_decay = _unwrap_singleton(cast(Optional[list[float]], sampling_config.top_p_decay))
    top_p_min = _unwrap_singleton(cast(Optional[list[float]], sampling_config.top_p_min))
    top_p_reset_ids = _unwrap_singleton(cast(Optional[list[int]], sampling_config.top_p_reset_ids))
    beam_width_out = _get_beam_width_out(request)
    beam_width_in = _get_beam_width_in(request)
    use_beam_search = _get_max_beam_width(request) > 1

    return UtilsSamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        beam_width_in=beam_width_in,
        beam_width_out=beam_width_out,
        use_beam_search=use_beam_search,
        top_p_decay=top_p_decay,
        top_p_min=top_p_min,
        top_p_reset_ids=top_p_reset_ids,
    )


def _request_sampling_params_cachable(params: UtilsSamplingParams) -> bool:
    return not params.use_beam_search


def _request_strategy(request: LlmRequest, *, vocab_size: int) -> Strategy:
    # We try to cache the resolved strategy on the request object, as it's not cheap enough to
    # resolve it on every iteration.
    cached_sampling_strategy = request.py_sampling_strategy
    if cached_sampling_strategy is not None:
        return cached_sampling_strategy

    params = _request_get_sampling_params(request)
    sampling_strategy = resolve_sampling_strategy(params, vocab_size=vocab_size)
    if _request_sampling_params_cachable(params):
        request.py_sampling_strategy = sampling_strategy
    return sampling_strategy
