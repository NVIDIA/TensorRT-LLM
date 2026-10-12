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

"""Draft-token verification for two-model speculative decoding.

Only reached when the drafter runs as a separate engine (NGram, two-model
Eagle3/draft-target). One-model speculation (MTP / Eagle3 one-model / SA)
verifies via ``SpecSampler`` and never enters this path -- see the note in
``speculative/drafter.py`` on which drafters use ``TorchSampler``.

Draft tokens are accepted while they match the tokens the target sampled at
the same positions. ``TorchSampler`` owns one :class:`TwoModelSpecDecHandler`
and delegates to :meth:`TwoModelSpecDecHandler.process_draft_tokens`.
"""

from typing import TYPE_CHECKING

from ..llm_request import LlmRequest, get_draft_token_length
from .sampler_common import DEFAULT_BEAM_IDX, FinishReasonsList, add_token

if TYPE_CHECKING:
    from .sampler import TorchSampler

__all__ = ["TwoModelSpecDecHandler"]


class TwoModelSpecDecHandler:
    """Verifies draft tokens produced by a separate drafter engine.

    Holds a back-reference to the owning ``TorchSampler`` because verification
    is interleaved with its finish-reason bookkeeping (``finish_if_reason``).
    """

    def __init__(self, sampler: "TorchSampler"):
        self._sampler = sampler

    def process_draft_tokens(
        self,
        request: LlmRequest,
        new_tokens_list: list[list[list[int]]],
        finish_reasons: FinishReasonsList,
    ) -> int:
        """Add the target's tokens up to the first rejected draft token.

        Returns the number of accepted draft tokens.
        """
        new_token = add_token(request, new_tokens_list, beam_idx=DEFAULT_BEAM_IDX)
        stop = self._sampler.finish_if_reason(
            request, finish_reasons, step=0, beam_idx=DEFAULT_BEAM_IDX
        )
        if stop or get_draft_token_length(request) == 0:
            return 0
        num_accepted = 0

        if self._sampler._force_num_accepted_tokens != 0:
            # Force acceptance of up to force_num_accepted_tokens draft tokens
            force_limit = min(
                self._sampler._force_num_accepted_tokens, len(request.py_draft_tokens)
            )
            for _ in request.py_draft_tokens[:force_limit]:
                num_accepted += 1
                new_token = add_token(
                    request, new_tokens_list, beam_idx=DEFAULT_BEAM_IDX, step=num_accepted
                )
                if self._sampler.finish_if_reason(
                    request, finish_reasons, step=num_accepted, beam_idx=DEFAULT_BEAM_IDX
                ):
                    break
        else:
            for draft_token in request.py_draft_tokens:
                if draft_token != new_token:
                    # Reject.
                    break

                num_accepted += 1
                new_token = add_token(
                    request, new_tokens_list, beam_idx=DEFAULT_BEAM_IDX, step=num_accepted
                )
                if self._sampler.finish_if_reason(
                    request, finish_reasons, step=num_accepted, beam_idx=DEFAULT_BEAM_IDX
                ):
                    break
        return num_accepted
