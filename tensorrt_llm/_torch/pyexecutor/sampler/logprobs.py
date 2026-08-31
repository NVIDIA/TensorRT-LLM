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

"""Log-probs handling for ``TorchSampler``.

Holds the device-side log-probs buffers (:class:`LogProbsStore`), the staged
host copies (:class:`LogProbsState` / :class:`LogProbsStateList`), the
conversions from those into the per-request result format, and the feature's
whole per-step lifecycle in :class:`LogProbsHandler`; ``TorchSampler`` owns one
instance and drives it through batch preparation and the per-step gather.

The handler also owns the top-k sizing state -- ``max_topk_logprobs`` and the
derived ``TOPK_LOGPROBS_SHAPE`` -- which it grows in place when a batch asks
for more top-k slots than the buffers currently hold. ``TorchSampler`` reads
that shape when it allocates the store, so the handler is constructed first.

Beam search keeps its own log-prob path: it accumulates a whole beam's history
and emits it at :func:`beam_search.finalize_beam`, sharing only the
:class:`LogProbsStore` buffers with the per-step path here.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

import torch

from tensorrt_llm._utils import nvtx_range, prefer_pinned
from tensorrt_llm.executor.result import Logprob, SimpleTokenLogprobs, TokenLogprobs
from tensorrt_llm.sampling_params import MAX_TOP_LOGPROBS, check_logprobs_limit

from ..llm_request import LlmRequest
from .ops.vanilla import Fusions
from .sampler_common import _BatchedSamplingResult
from .sampler_features import _UnpackedStepIndexer

if TYPE_CHECKING:
    from .sampler import TorchSampler

__all__ = [
    "LogProbsState",
    "LogProbsStateList",
    "LogProbsStore",
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


class LogProbsHandler:
    """Owns the log-probs sizing state and the per-step log-probs work.

    ``TorchSampler`` holds one instance. The top-k buffers are sized by
    ``max_topk_logprobs``, which grows on demand in :meth:`prepare` when a batch
    asks for more than the current width; ``TOPK_LOGPROBS_SHAPE`` follows it and
    is read by the sampler when it allocates the store.

    Holds a back-reference to the sampler for the batch-shape scalars
    (``max_num_sequences`` / ``max_tokens`` / ``max_beam_width``) and its
    ``store``, which the log-probs tensors are slices of.
    """

    def __init__(self, sampler: "TorchSampler"):
        self._sampler = sampler
        self.max_topk_logprobs = MAX_TOP_LOGPROBS
        self.batch_max_topk_logprobs = 0
        self.TOPK_LOGPROBS_SHAPE = (
            sampler.max_num_sequences,
            sampler.max_tokens,
            self.max_topk_logprobs,
        )

    def handle_logprobs(
        self,
        request: LlmRequest,
        logprobs_state_list: LogProbsStateList | None,
        *,
        count: int,
    ) -> None:
        if request.py_return_log_probs:
            beam_width = request.py_beam_width
            assert request.py_num_logprobs is not None, "request.py_num_logprobs must be provided"
            assert logprobs_state_list is not None, "logprobs_state_list must be provided"
            assert request.py_seq_slot is not None
            token_log_probs = store_logprobs_list_to_request(
                logprobs_state_list,
                request.py_seq_slot,
                beam_width,
                count,
                request.py_num_logprobs,
                simple_format=request.py_logprobs_simple_format,
            )
            request.py_result.append_log_probs(token_log_probs)

    def _return_log_probs(self, requests: list[LlmRequest]) -> bool:
        return any(req.py_return_log_probs for req in requests)

    def _prepare_log_probs(self, requests: list[LlmRequest]) -> None:
        self.batch_max_topk_logprobs = max(
            (req.py_num_logprobs or 0 for req in requests),
            default=0,
        )
        check_logprobs_limit("batch_max_logprobs", self.batch_max_topk_logprobs, MAX_TOP_LOGPROBS)
        if self.max_topk_logprobs < self.batch_max_topk_logprobs:
            self.max_topk_logprobs = self.batch_max_topk_logprobs
            self.TOPK_LOGPROBS_SHAPE = (
                self._sampler.max_num_sequences,
                self._sampler.max_tokens,
                self.max_topk_logprobs,
            )
            log_probs_store = self._sampler.store.log_probs_store
            log_probs_store.topk_vals.resize_(self.TOPK_LOGPROBS_SHAPE)
            log_probs_store.topk_indices.resize_(self.TOPK_LOGPROBS_SHAPE)

    @nvtx_range("_process_logprobs")
    def _process_logprobs(
        self,
        batched_sampling_result: _BatchedSamplingResult,
        *,
        logits_cuda: torch.Tensor,
        new_tokens_cuda: torch.Tensor,
        seq_slots: torch.Tensor,
        requests: list[LlmRequest],
        req_num_generated_tokens: torch.Tensor,
    ) -> None:
        logprobs_cuda = batched_sampling_result.logprobs_cuda
        assert logprobs_cuda is not None  # _process_logprobs call is gated by return_log_probs

        raw_logprobs_reqs_indices = batched_sampling_result.raw_logprobs_reqs_indices
        if raw_logprobs_reqs_indices:
            # Insert raw logprobs into logprobs_cuda.
            #
            # NB: Cannot reuse softmax from _sample_batched_by_strategy, because raw logprobs are specified
            #     to correspond to temperature=1.
            raw_logprobs_logit_indices_cuda = (
                batched_sampling_result.raw_logprobs_logit_indices_cuda
            )
            assert raw_logprobs_logit_indices_cuda is not None
            raw_logprobs_start = batched_sampling_result.processed_logprobs_end
            raw_logprobs_end = raw_logprobs_start + raw_logprobs_logit_indices_cuda.size(0)
            # NB: There is no separate code path resolving contiguous ranges to 'slice', because the performance
            #     impact after kernel fusion is anticipated to be small (raw_logprobs_logit_indices_cuda is sorted).
            Fusions.gather_log_softmax_with_output(
                logits_cuda,
                raw_logprobs_logit_indices_cuda,
                out=logprobs_cuda[raw_logprobs_start:raw_logprobs_end],
            )
            logprobs_end = raw_logprobs_end
        else:
            logprobs_end = batched_sampling_result.processed_logprobs_end

        # Process raw and processed logprobs jointly from here on
        logprobs_reqs_indices = (
            batched_sampling_result.processed_logprobs_reqs_indices + raw_logprobs_reqs_indices
        )
        logprobs_cuda = logprobs_cuda[:logprobs_end]

        # NB: The amount of data copied into logprobs_cuda could be reduced by performing the
        #     sampled-token / top-k selection earlier, since most logprobs are discarded when
        #     returning only sampled-token logprobs / top-k logprobs.

        log_probs_store = self._sampler.store.log_probs_store

        if logprobs_reqs_indices:
            logprobs_reqs_indices_1_beam = []
            logprobs_reqs_indices_n_beam = []
            for req_idx in logprobs_reqs_indices:
                if requests[req_idx].py_beam_width == 1:
                    logprobs_reqs_indices_1_beam.append(req_idx)
                else:
                    logprobs_reqs_indices_n_beam.append(req_idx)

            slot_and_step_size = new_tokens_cuda.size(0) * new_tokens_cuda.size(1)

            def _gather_src_dst_indices(
                reqs_indices_tensor: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                # Gather indices for new_tokens_cuda
                # NB: Not reusing indexer from _unbatch_sampling_results in order to not add work
                #     in case logprobs are not requested.
                seq_slots_selection = seq_slots[reqs_indices_tensor]
                req_num_generated_tokens_selection = req_num_generated_tokens[reqs_indices_tensor]
                src_indices_cuda = _UnpackedStepIndexer(
                    seq_slots=seq_slots_selection,
                    num_steps=req_num_generated_tokens_selection,
                    steps_dim_size=new_tokens_cuda.size(0),
                    slots_dim_size=new_tokens_cuda.size(1),
                    dim_order=_UnpackedStepIndexer.DimOrder.STEP_MAJOR,
                )[:].to(
                    device=logits_cuda.device,
                    non_blocking=True,
                )
                # Scatter indices for logprobs storage
                # NB: Would not be necessary if new_tokens_cuda and logprobs storage tensors shared
                #     a common layout.
                dst_indices_cuda = _UnpackedStepIndexer(
                    seq_slots=seq_slots_selection,
                    num_steps=req_num_generated_tokens_selection,
                    steps_dim_size=new_tokens_cuda.size(0),
                    slots_dim_size=new_tokens_cuda.size(1),
                    dim_order=_UnpackedStepIndexer.DimOrder.SLOT_MAJOR,
                    index_dtype=torch.int64,  # enforced by Tensor.scatter_
                )[:].to(
                    device=logits_cuda.device,
                    non_blocking=True,
                )
                return src_indices_cuda, dst_indices_cuda

            if logprobs_reqs_indices_1_beam:
                logprobs_reqs_indices_1_beam_tensor = torch.tensor(
                    logprobs_reqs_indices_1_beam, dtype=torch.int32
                )

                src_indices_cuda, dst_indices_cuda = _gather_src_dst_indices(
                    logprobs_reqs_indices_1_beam_tensor
                )

                # Squash beams dimension
                sampled_log_prob_indices = log_probs_store.sampled_log_prob_indices[:, 0, :]
                sampled_log_prob_ranks = log_probs_store.sampled_log_prob_ranks[:, 0, :]
                sampled_log_probs = log_probs_store.sampled_log_probs[:, 0, :]
                new_tokens_cuda_1_beam = new_tokens_cuda[..., 0]

                assert sampled_log_probs.transpose(0, 1).shape == new_tokens_cuda_1_beam.shape
                assert (
                    sampled_log_prob_indices.transpose(0, 1).shape == new_tokens_cuda_1_beam.shape
                )
                assert sampled_log_prob_ranks.transpose(0, 1).shape == new_tokens_cuda_1_beam.shape

                # Gather sampled tokens / logprobs indices
                sampled_indices_cuda = new_tokens_cuda_1_beam.view(slot_and_step_size).gather(
                    dim=0, index=src_indices_cuda
                )

                # Get the sampled logprobs
                # NB: logprobs_cuda contains logprobs only for the single-beam requests, since beam search handles
                #     logprobs elsewhere.
                sampled_vals_cuda = torch.gather(
                    logprobs_cuda, dim=1, index=sampled_indices_cuda.unsqueeze(-1)
                ).squeeze(-1)  # flattened (step, slot)

                # sampled_rank_cuda contains the 0-based rank, it will be corrected to 1-based in handle_logprobs
                # NB: Computation of sampled rank could be lowered into FlashInferGroupedStrategySampler, s.t., e.g.,
                #     for greedy sampling, logits management and log_softmax could be completely skipped (sampled rank
                #     computation is trivial in this case).
                sampled_rank_cuda = Fusions.determine_sampled_rank(
                    logprobs_cuda,
                    sampled_vals_cuda.unsqueeze(-1),
                )

                sampled_log_prob_indices.view(slot_and_step_size).scatter_(
                    dim=0, index=dst_indices_cuda, src=sampled_indices_cuda
                )
                sampled_log_probs.view(slot_and_step_size).scatter_(
                    dim=0, index=dst_indices_cuda, src=sampled_vals_cuda
                )
                sampled_log_prob_ranks.view(slot_and_step_size).scatter_(
                    dim=0, index=dst_indices_cuda, src=sampled_rank_cuda
                )

                # Process the topk logprobs
                if self.batch_max_topk_logprobs > 0:
                    # Get the topk logprobs
                    topk_vals_cuda, topk_indices_cuda = torch.topk(
                        logprobs_cuda,
                        k=self.batch_max_topk_logprobs,
                        dim=-1,
                    )

                    topk_expanded_indices_cuda = dst_indices_cuda.view(-1, 1).expand(
                        -1, topk_vals_cuda.size(-1)
                    )
                    log_probs_store.topk_vals[..., : self.batch_max_topk_logprobs].view(
                        self._sampler.max_num_sequences * self._sampler.max_tokens,
                        self.batch_max_topk_logprobs,
                    ).scatter_(dim=0, index=topk_expanded_indices_cuda, src=topk_vals_cuda)
                    log_probs_store.topk_indices[..., : self.batch_max_topk_logprobs].view(
                        self._sampler.max_num_sequences * self._sampler.max_tokens,
                        self.batch_max_topk_logprobs,
                    ).scatter_(
                        dim=0,
                        index=topk_expanded_indices_cuda,
                        src=topk_indices_cuda.to(torch.int32),
                    )

            # Because req_num_generated_tokens may differ from the number of sampled tokens in
            # beam search, the sampled rank computation would entail extra complexity to resolve
            # the relationships between incoming and outgoing beams. For the sampled logprobs, this
            # matching happens in beam_search_sampling_batch_cba() which updates
            # log_probs_store.sampled_log_probs. Therefore, neither sampled ranks nor sampled logprobs
            # are handled here.
            if logprobs_reqs_indices_n_beam:
                logprobs_reqs_indices_n_beam_tensor = torch.tensor(
                    logprobs_reqs_indices_n_beam, dtype=torch.int32
                )

                src_indices_cuda, dst_indices_cuda = _gather_src_dst_indices(
                    logprobs_reqs_indices_n_beam_tensor
                )

                # NB: The transpose only works (yields contiguous tensors) if self._sampler.max_tokens=1 and would
                #     not be necessary, if the code was refactored such that
                #     LOGPROBS_SHAPE = (self._sampler.max_num_sequences,
                #                       self._sampler.max_tokens, self._sampler.max_beam_width)
                sampled_log_prob_indices = log_probs_store.sampled_log_prob_indices.transpose(1, 2)
                assert sampled_log_prob_indices.transpose(0, 1).shape == new_tokens_cuda.shape

                logprobs_inout_indices_cuda_size = src_indices_cuda.size(0)

                # Gather sampled tokens / logprobs indices
                beam_expanded_src_indices_cuda = src_indices_cuda.unsqueeze(-1).expand(
                    logprobs_inout_indices_cuda_size, self._sampler.max_beam_width
                )
                sampled_indices_cuda = new_tokens_cuda.view(
                    slot_and_step_size, self._sampler.max_beam_width
                ).gather(dim=0, index=beam_expanded_src_indices_cuda)

                beam_expanded_dst_indices_cuda = dst_indices_cuda.unsqueeze(-1).expand(
                    logprobs_inout_indices_cuda_size, self._sampler.max_beam_width
                )
                sampled_log_prob_indices.view(
                    slot_and_step_size, self._sampler.max_beam_width
                ).scatter_(dim=0, index=beam_expanded_dst_indices_cuda, src=sampled_indices_cuda)
