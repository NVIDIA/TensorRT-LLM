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

"""Log-probs state and conversion helpers for ``TorchSampler``.

Holds the device-side log-probs buffers (:class:`LogProbsStore`), the staged
host copies (:class:`LogProbsState` / :class:`LogProbsStateList`), and the
pure conversions from those into the per-request result format.

Sizing and the per-step device gather stay in ``TorchSampler``: they depend on
the sampler-wide shapes (``TOPK_LOGPROBS_SHAPE`` and friends) that the sampler
grows in place as batches demand more top-k slots.
"""

from dataclasses import dataclass
from typing import TypeAlias, cast

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.executor.result import Logprob, SimpleTokenLogprobs, TokenLogprobs

from ..llm_request import LlmRequest

__all__ = [
    "LogProbsState",
    "LogProbsStateList",
    "LogProbsStore",
    "convert_logprobs_tensor_to_list",
    "get_logprobs_from_request",
    "store_logprobs_list_to_request",
]


@dataclass(kw_only=True)
class LogProbsState:
    sampled_vals: torch.Tensor
    sampled_indices: torch.Tensor
    sampled_rank: torch.Tensor
    topk_vals: torch.Tensor
    topk_indices: torch.Tensor


_LogProbsFloatState: TypeAlias = list[list[list[float]]]
_LogProbsIntState: TypeAlias = list[list[list[int]]]


@dataclass(kw_only=True)
class LogProbsStateList:
    sampled_vals: _LogProbsFloatState
    sampled_indices: _LogProbsIntState
    sampled_rank: _LogProbsIntState
    topk_vals: _LogProbsFloatState
    topk_indices: _LogProbsIntState

    @staticmethod
    def from_logprobs_state(logprobs_state: LogProbsState) -> "LogProbsStateList":
        return LogProbsStateList(
            sampled_vals=logprobs_state.sampled_vals.tolist(),
            sampled_indices=logprobs_state.sampled_indices.tolist(),
            topk_vals=logprobs_state.topk_vals.tolist(),
            topk_indices=logprobs_state.topk_indices.tolist(),
            sampled_rank=logprobs_state.sampled_rank.tolist(),
        )


@dataclass(kw_only=True)
class LogProbsStore:
    """Auxiliary data structures used for log-probs handling."""

    sampled_log_prob_indices: torch.Tensor
    """Shape: batch_size, beam_width, max_tokens
       Usage: Stores the token indices of the sampled logprobs"""
    sampled_log_probs: torch.Tensor
    """Shape: batch_size, beam_width, max_tokens
       Usage: Stores the values of the sampled logprobs"""
    sampled_log_prob_ranks: torch.Tensor
    """Shape: batch_size, beam_width, max_tokens
       Usage: Stores the ranks of the sampled logprobs"""
    topk_indices: torch.Tensor
    """Shape: batch_size, max_tokens, max_topk_logprobs
       Usage: Stores the token indices of the topk logprobs"""
    topk_vals: torch.Tensor
    """Shape: batch_size, max_tokens, max_topk_logprobs
       Usage: Stores the values of the topk logprobs"""


def convert_logprobs_tensor_to_list(
    token_tensor: torch.Tensor,
    logprobs_tensor: torch.Tensor,
) -> list[list[dict[int, Logprob]]]:
    """Convert the logprobs tensor to a list of lists of dictionaries of Logprob objects

    Logprobs storage expects logprobs as a list[list[dict[int, Logprob]]] object

    args:
        token_tensor: torch.Tensor. Shape: beam_width, num_tokens, num_logprobs
        logprobs_tensor: torch.Tensor. Shape: beam_width, num_tokens, num_logprobs
    output:
        list[list[dict[int, Logprob]]]. Shape: (beam_width, num_tokens)
    """
    assert token_tensor.dim() == 3 and logprobs_tensor.dim() == 3, (
        f"Token and logprobs tensors must have 3 dimensions (beam_width, num_tokens, num_logprobs). \
        Got shapes (token_tensor) {token_tensor.shape} and (logprobs_tensor) {logprobs_tensor.shape} instead"
    )

    token_log_probs: list[list[dict[int, Logprob]]] = []
    token_list = token_tensor.tolist()
    logprobs_list = logprobs_tensor.tolist()
    for beam_idx in range(token_tensor.shape[0]):
        beam_token_log_probs: list[dict[int, Logprob]] = []
        for topk_token, topk_logprob in zip(token_list[beam_idx], logprobs_list[beam_idx]):
            logprobs = {
                token: Logprob(logprob=logprob, rank=rank + 1)
                for rank, (token, logprob) in enumerate(zip(topk_token, topk_logprob))
            }
            beam_token_log_probs.append(logprobs)
        token_log_probs.append(beam_token_log_probs)

    return token_log_probs


def store_logprobs_list_to_request(
    logprobs_state_list: LogProbsStateList,
    req_seq_slot: int,
    beam_width: int,
    count: int,
    num_topk_logprobs: int,
    simple_format: bool = False,
) -> list[list[dict[int, Logprob]]] | list[list[float]]:
    """Convert the LogProbsStateList object to per-token logprobs.

    By default returns ``list[list[dict[int, Logprob]]]``. When
    ``simple_format`` is True and ``num_topk_logprobs == 0`` the result is a
    flat ``list[list[float]]`` (one logprob per generated token, per beam).

    args:
        logprobs_state_list: LogProbsStateList. Contains the topk indices, topk values,
            sampled indices, sampled values, and sampled ranks.
        req_seq_slot: int. The sequence slot of the request.
        beam_width: int. The beam width of the request.
        count: int. The number of tokens to store.
        num_topk_logprobs: int. The number of topk logprobs of each token.
        simple_format: bool. If True (and num_topk_logprobs == 0), return
            ``list[list[float]]`` instead of the dict format. Avoids per-token
            dict allocation when only the sampled-token logprob is needed.
    output:
        list[list[dict[int, Logprob]]] (default) or list[list[float]] (simple format).
        Shape: (beam_width, count)
    """

    sampled_log_probs_indices_list = logprobs_state_list.sampled_indices[req_seq_slot]
    sampled_log_probs_vals_list = logprobs_state_list.sampled_vals[req_seq_slot]
    sampled_log_probs_rank_list = logprobs_state_list.sampled_rank[req_seq_slot]

    if num_topk_logprobs == 0:
        if simple_format:
            token_log_probs_simple: list[list[float]] = [
                [sampled_log_probs_vals_list[beam_idx][step_idx] for step_idx in range(count)]
                for beam_idx in range(beam_width)
            ]
            return token_log_probs_simple

        token_log_probs: list[list[dict[int, Logprob]]] = [
            [
                {
                    sampled_log_probs_indices_list[beam_idx][step_idx]: Logprob(
                        sampled_log_probs_vals_list[beam_idx][step_idx],
                        sampled_log_probs_rank_list[beam_idx][step_idx] + 1,
                    )
                }
                for step_idx in range(count)
            ]
            for beam_idx in range(beam_width)
        ]
    else:
        token_list = logprobs_state_list.topk_indices[req_seq_slot]
        logprobs_list = logprobs_state_list.topk_vals[req_seq_slot]
        token_log_probs = [[] for _ in range(beam_width)]
        for step_idx in range(count):
            topk_tokens = token_list[step_idx][:num_topk_logprobs]
            topk_logprobs = logprobs_list[step_idx][:num_topk_logprobs]
            min_rank = len(topk_tokens) + 1

            topk_logprob_dict = {
                token: Logprob(logprob=logprob, rank=rank + 1)
                for rank, (token, logprob) in enumerate(zip(topk_tokens, topk_logprobs))
            }

            for beam_idx in range(beam_width):
                # NB: Keeps sampled token in the first position (cf. https://stackoverflow.com/a/67786863)
                logprobs = {
                    sampled_log_probs_indices_list[beam_idx][step_idx]: Logprob(
                        logprob=sampled_log_probs_vals_list[beam_idx][step_idx],
                        rank=max(
                            min_rank,
                            sampled_log_probs_rank_list[beam_idx][step_idx] + 1,
                        ),
                    ),
                    **topk_logprob_dict,
                }
                token_log_probs[beam_idx].append(logprobs)

    return token_log_probs


def get_logprobs_from_request(
    request: LlmRequest,
    pin_memory: bool = True,
    preallocate_extra_steps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the logprobs from the request.

    Returns:
        logprobs_tensor: A tensor of shape (beam_width, num_generated_tokens, num_logprobs)
        logprobs_indices_tensor: A tensor of shape (beam_width, num_generated_tokens, num_logprobs)
    """
    pin_memory = pin_memory and prefer_pinned()
    num_generated_tokens = request.max_beam_num_tokens - request.py_prompt_len
    assert request.py_num_logprobs == 0, (
        "Beam search only supports returning the sampled logprob per token"
    )
    logprobs_tensor_full = torch.empty(
        (
            request.py_beam_width,
            num_generated_tokens + preallocate_extra_steps,
            request.py_num_logprobs + 1,
        ),
        pin_memory=pin_memory,
        dtype=torch.float32,
    )
    logprobs_indices_tensor_full = torch.empty(
        (
            request.py_beam_width,
            num_generated_tokens + preallocate_extra_steps,
            request.py_num_logprobs + 1,
        ),
        pin_memory=pin_memory,
        dtype=torch.int32,
    )
    # NB: forward slicing, because [:, :-0, :] would yield an empty view
    #     instead of the full history when preallocate_extra_steps == 0.
    logprobs_tensor = logprobs_tensor_full[:, :num_generated_tokens, :]
    logprobs_indices_tensor = logprobs_indices_tensor_full[:, :num_generated_tokens, :]
    if logprobs_tensor.numel() > 0:
        logprobs_list = request.py_result.log_probs
        assert logprobs_list is not None

        if request.py_logprobs_simple_format:
            tokens = request.get_tokens()
            for beam_idx, beam_logprobs in enumerate(logprobs_list):
                beam_logprobs = cast(SimpleTokenLogprobs, beam_logprobs)
                for token_idx, token_logprobs_simple in enumerate(beam_logprobs):
                    logprobs_tensor[beam_idx, token_idx, 0] = token_logprobs_simple
                    logprobs_indices_tensor[beam_idx, token_idx, 0] = tokens[beam_idx][token_idx]
        else:
            for beam_idx, beam_logprobs in enumerate(logprobs_list):
                beam_logprobs = cast(TokenLogprobs, beam_logprobs)
                for token_idx, token_logprobs in enumerate(beam_logprobs):
                    for key, value in token_logprobs.items():
                        assert value.rank is not None
                        logprobs_tensor[beam_idx, token_idx, value.rank - 1] = value.logprob
                        logprobs_indices_tensor[beam_idx, token_idx, value.rank - 1] = key
    return logprobs_tensor_full, logprobs_indices_tensor_full
