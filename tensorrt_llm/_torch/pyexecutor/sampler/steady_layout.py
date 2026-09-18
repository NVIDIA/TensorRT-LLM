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

"""Batch-layout caches of ``TorchSampler``'s host path.

Consecutive single-step decode iterations over the same requests in the same order
differ only in the logits and the sequence lengths; every index tensor the sampler
derives from the request list (sequence slots, request offsets, strategy groups,
per-group logits indices, logprob gather / scatter indices, stop-word masks, result
scatter indices) is identical. :class:`_SteadyDecodeLayout` records them on the first
step of a new layout so later steps reuse them; :class:`_GroupIndexTensors` and
:class:`_LogprobsIndexTensors` hold the per-strategy-group and log-probs parts.
:class:`_IdentityStepIndexer` is the index map of a batch with one token and one beam
per request (row ``i`` <-> request ``i``).

Plain data with no sampler imports at runtime, so the feature modules that consume
the caches (``finish_reasons``, ``logprobs``) and the orchestration in ``sampler``
can all import it.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import torch

from tensorrt_llm._utils import prefer_pinned

if TYPE_CHECKING:
    from .sampler import SamplingRequestsMetadata
    from .sampler_strategy import Strategy

__all__ = [
    "_GroupIndexTensors",
    "_IdentityStepIndexer",
    "_LogprobsIndexTensors",
    "_SteadyDecodeLayout",
]


class _IdentityStepIndexer:
    """``_PackedStepIndexer`` for batches of one step and one beam per request
    (no speculative decoding, no beam search): with dense packing the token of
    request ``i`` is row ``i``, so the index map is ``arange`` and no per-call
    step / offset tensors are needed. Backed by a constant int64 buffer (the
    dtype the packed indexer derives from its cumsum offsets)."""

    def __init__(self, arange_host: torch.Tensor, num_requests: int):
        self._index_map = arange_host[:num_requests]

    def __getitem__(self, req_indices: Any) -> torch.Tensor:
        return self._index_map[req_indices].view(-1)


@dataclass
class _GroupIndexTensors:
    """Per-strategy-group data of ``_sample_batched_by_strategy`` that only depends on the
    batch layout (request order, steps per request, logprob flags), not on the step's
    logits."""

    strategies_per_step: list["Strategy"]
    steps_per_request: list[int]
    # Contiguous groups slice the batch logits; others index them on device.
    sampler_slice: Optional[slice]
    group_logits_cuda_indices_cuda: Optional[torch.Tensor]
    # Processed logprobs: the requests of the group asking for them (batch indices) ...
    processed_req_indices: list[int] = field(default_factory=list)
    # ... how many of them are gathered here (beam-search requests are excluded) ...
    num_gather_processed: int = 0
    # ... and, when only part of the group is gathered, the per-step selection mask and
    # its count (the logits indices of the selected rows when the group is not contiguous).
    proc_lp_step_mask_cuda: Optional[torch.Tensor] = None
    proc_lp_steps_num_selected: int = 0
    proc_lp_logits_indices_cuda: Optional[torch.Tensor] = None


@dataclass
class _LogprobsIndexTensors:
    """Index tensors of ``LogProbsHandler._process_logprobs`` for a fixed batch layout."""

    reqs_indices_1_beam: list[int]
    reqs_indices_n_beam: list[int]
    src_indices_1_beam_cuda: Optional[torch.Tensor] = None
    dst_indices_1_beam_cuda: Optional[torch.Tensor] = None
    src_indices_n_beam_cuda: Optional[torch.Tensor] = None
    dst_indices_n_beam_cuda: Optional[torch.Tensor] = None


@dataclass
class _SteadyDecodeLayout:
    """Batch-layout cache of ``TorchSampler.sample_async`` for consecutive single-step
    decode iterations over the same requests in the same order.

    Between two such iterations only the logits and the sequence lengths change; every
    index tensor the sampler derives from the request list (seq slots, request offsets,
    strategy groups, logprob gather/scatter indices, stop-word masks, result scatter
    indices) is identical, yet the generic path rebuilds and re-uploads all of them on
    every step (~40 % of the sampler's host time at small batches). The first step of a
    new layout records them here; later steps reuse them and advance the sequence lengths
    in place. Invalidated by any change of the (request id, sequence slot) tuple
    (``signature``); never used for context, draft, beam-search or top-p-decay batches.
    """

    signature: tuple
    num_requests: int
    disabled: bool = False
    metadata: Optional["SamplingRequestsMetadata"] = None
    return_log_probs: bool = False
    batch_max_topk_logprobs: int = 0
    seq_slots_host: Optional[torch.Tensor] = None
    seq_slots_cuda: Optional[torch.Tensor] = None
    # Host copy of the sequence lengths: a plain (non-pinned) tensor advanced in place;
    # the pinned tensor that seeded the device copy is never written again.
    seq_lens_host: Optional[torch.Tensor] = None
    seq_lens_cuda: Optional[torch.Tensor] = None
    first_seq_len: int = -1
    last_seq_len: int = -1
    # Strategy groups (with metadata) and the raw-logprobs mask of the grouper.
    grouped: Optional[dict] = None
    need_raw_logprobs: Optional[torch.Tensor] = None
    batch_req_indices: Optional[torch.Tensor] = None
    group_index: dict = field(default_factory=dict)
    # Raw logprobs: the requests (batch indices) and their logits rows.
    raw_logprobs_reqs_indices: Optional[list[int]] = None
    raw_logprobs_logit_indices_cuda: Optional[torch.Tensor] = None
    logprobs_index: Optional[_LogprobsIndexTensors] = None
    batch_dest_indices_1d_cuda: Optional[torch.Tensor] = None
    stop_words_prep: Optional[tuple] = None

    def record_batch(
        self,
        metadata: "SamplingRequestsMetadata",
        return_log_probs: bool,
        batch_max_topk_logprobs: int,
        seq_slots_host: torch.Tensor,
        seq_slots_cuda: torch.Tensor,
        seq_lens_host: torch.Tensor,
        seq_lens_cuda: torch.Tensor,
    ) -> None:
        self.metadata = metadata
        self.return_log_probs = return_log_probs
        self.batch_max_topk_logprobs = batch_max_topk_logprobs
        self.seq_slots_host = seq_slots_host
        self.seq_slots_cuda = seq_slots_cuda
        self.seq_lens_host = seq_lens_host.clone()
        self.seq_lens_cuda = seq_lens_cuda
        lens = self.seq_lens_host
        self.first_seq_len = int(lens[0])
        self.last_seq_len = int(lens[-1])

    def advance_seq_lens(self, requests: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Sequence lengths for this step: +1 per request in the steady state (checked on
        the first and last request), rebuilt from the requests otherwise."""
        assert self.seq_lens_host is not None and self.seq_lens_cuda is not None
        first = requests[0].max_beam_num_tokens
        last = requests[-1].max_beam_num_tokens
        if first == self.first_seq_len + 1 and last == self.last_seq_len + 1:
            self.seq_lens_host.add_(1)
            self.seq_lens_cuda.add_(1)
        else:
            # Not the +1 regime (e.g. a request was rewound): rebuild like the
            # generic path, from a pinned staging tensor that is never written again.
            pinned = torch.tensor(
                [r.max_beam_num_tokens for r in requests],
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
            )
            self.seq_lens_cuda = pinned.to(device="cuda", non_blocking=True)
            self.seq_lens_host = pinned.clone()
        self.first_seq_len = first
        self.last_seq_len = last
        return self.seq_lens_host, self.seq_lens_cuda
