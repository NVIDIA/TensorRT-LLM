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

"""How a request's sampling params map onto the fused sampling kernel.

:func:`resolve_row_params` is the one interpretation of ``temperature`` /
``top_k`` / ``top_p`` / ``min_p`` shared by ``TorchSampler`` and the one-model
speculative path. :class:`SlotSamplingParams` holds its result per sequence
slot for ``TorchSampler``, written once at admission and gathered by slot on
every step.
"""

from enum import IntEnum
from typing import NamedTuple, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.sampling_params import LogprobMode, SamplingParams

from ..llm_request import LlmRequest
from .sampler_common import _request_get_sampling_params

__all__ = ["RowSamplingParams", "SamplingMode", "SlotSamplingParams", "resolve_row_params"]


class SamplingMode(IntEnum):
    GREEDY = 0
    SAMPLING = 1
    BEAM = 2


class RowSamplingParams(NamedTuple):
    """A request's per-row params for the fused kernel.

    Disabled filters take the kernel's neutral values: top_k 0, top_p 1.0,
    min_p 0.0. Greedy and beam-search requests carry neutral values throughout;
    beam search keeps its own params (``beam_search.BeamSearchParams``).
    """

    mode: SamplingMode
    temperature: float
    top_k: int
    top_p: float
    min_p: float


def resolve_row_params(
    *,
    temperature: Optional[float],
    top_k: Optional[int],
    top_p: Optional[float],
    min_p: Optional[float],
    top_p_decay: Optional[float] = None,
    use_beam_search: bool = False,
) -> RowSamplingParams:
    """Resolve a request's sampling params, as specified on ``SamplingParams``.

    The greedy verdict, including the top-p decay override of the implicit
    greedy default, is ``SamplingParams.params_imply_greedy_decoding``. A
    ``top_k`` at or above the vocab size needs no special case: the kernel
    treats it as disabled.
    """
    if use_beam_search:
        return RowSamplingParams(SamplingMode.BEAM, 1.0, 0, 1.0, 0.0)
    if SamplingParams.params_imply_greedy_decoding(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_beam_search=False,
        min_p=min_p,
        top_p_decay=top_p_decay,
    ):
        return RowSamplingParams(SamplingMode.GREEDY, 1.0, 0, 1.0, 0.0)

    # Not greedy, so a set temperature is non-zero, top_k is not 1 and top_p is
    # not 0.
    temperature = temperature or 1.0
    top_k = top_k or 0
    top_p = top_p or 1.0
    assert temperature > 0, f"non-greedy sampling needs temperature > 0, got {temperature}"
    assert top_k == 0 or top_k > 1, f"non-greedy sampling needs a valid top_k, got {top_k}"
    assert top_p > 0, f"non-greedy sampling needs top_p > 0, got {top_p}"
    min_p = min_p if min_p is not None and min_p > 0.0 else 0.0
    return RowSamplingParams(SamplingMode.SAMPLING, temperature, top_k, top_p, min_p)


def request_row_params(request: LlmRequest) -> RowSamplingParams:
    params = _request_get_sampling_params(request)
    return resolve_row_params(
        temperature=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        min_p=params.min_p,
        top_p_decay=params.top_p_decay,
        use_beam_search=bool(params.use_beam_search),
    )


class SlotSamplingParams:
    """``TorchSampler``'s resolved sampling params, indexed by sequence slot.

    The host keeps each slot's mode and log-prob flags, which decide how a
    request is batched; the device keeps the fused kernel's per-row params.
    A slot that no request has been admitted to reads as greedy with neutral
    params, so every row is a valid kernel input; the last slot is the CUDA
    graph scratch row and is never admitted.
    """

    def __init__(self, num_slots: int, device: torch.device | str):
        self.modes: list[SamplingMode] = [SamplingMode.GREEDY] * num_slots
        self.need_processed_logprobs: list[bool] = [False] * num_slots
        self.need_raw_logprobs: list[bool] = [False] * num_slots
        # (temperature, top_p, min_p), so admission and a gather move them together.
        self.floats = torch.tensor(
            [[1.0], [1.0], [0.0]], dtype=torch.float32, device=device
        ).repeat(1, num_slots)
        self.top_k = torch.zeros(num_slots, dtype=torch.int32, device=device)

    def admit(self, requests: list[LlmRequest], slots_cuda: torch.Tensor) -> None:
        """Resolve newly admitted requests into their slots (int64 ``slots_cuda``)."""
        rows = [request_row_params(request) for request in requests]
        for request, row in zip(requests, rows):
            slot = request.py_seq_slot
            assert slot is not None
            self.modes[slot] = row.mode
            return_log_probs = request.return_log_probs
            self.need_processed_logprobs[slot] = (
                return_log_probs and request.py_logprobs_mode == LogprobMode.PROCESSED
            )
            self.need_raw_logprobs[slot] = (
                return_log_probs and request.py_logprobs_mode == LogprobMode.RAW
            )
        _, temperatures, top_ks, top_ps, min_ps = zip(*rows)
        pin = prefer_pinned()
        floats = torch.tensor([temperatures, top_ps, min_ps], dtype=torch.float32, pin_memory=pin)
        top_k = torch.tensor(top_ks, dtype=torch.int32, pin_memory=pin)
        device = self.floats.device
        self.floats.index_copy_(1, slots_cuda, floats.to(device, non_blocking=True))
        self.top_k.index_copy_(0, slots_cuda, top_k.to(device, non_blocking=True))

    def gather(
        self, slots_cuda: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-row ``(temperature, top_k, top_p, min_p)`` of the rows' slots."""
        floats = self.floats.index_select(1, slots_cuda)
        return floats[0], self.top_k.index_select(0, slots_cuda), floats[1], floats[2]
