# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Composable SA (Suffix Automaton) draft enhancer for one-engine speculative decoding workers.

When enabled, SA pattern matching overrides neural draft tokens for requests
where the suffix match length exceeds the configured threshold.
"""

from typing import List, Optional

import torch

from .suffix_automaton import SuffixAutomatonManager


class SADraftEnhancer:
    """Composable SA enhancement for any one-engine spec worker.

    This class encapsulates all SA-specific logic (extend, prepare buffers,
    override draft tokens) so that any worker (MTP, EAGLE3, PARD, etc.) can
    opt into SA enhancement.

    Usage:
        1. Construct once during worker ``__init__`` when ``use_sa_spec`` is True.
        2. Call ``extend_and_prepare`` after ``sample_and_accept_draft_tokens``.
        3. Call ``maybe_override_all_draft_tokens`` once after all draft layers
           have finished, so that neural draft layers never see SA tokens.
    """

    def __init__(self, sa_spec_threshold: int):
        self.sa_spec_threshold = sa_spec_threshold
        self.sa_match_len: Optional[torch.Tensor] = None
        self.sa_draft_tokens: Optional[torch.Tensor] = None
        self.sa_spec_index: int = 0

    def extend_and_prepare(
        self,
        sa_manager: SuffixAutomatonManager,
        request_ids: List[int],
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_gens: int,
        num_contexts: int,
        max_draft_len: int,
    ) -> None:
        """Extend SA states with accepted tokens and prepare override buffers.

        Must be called after ``sample_and_accept_draft_tokens`` and before the
        draft generation loop.

        Args:
            sa_manager: The SuffixAutomatonManager instance.
            request_ids: Full request ID list (contexts + generations).
            accepted_tokens: [batch_size, padded_width] accepted tokens
                (may be wider than max_draft_len + 1 due to caller padding).
            num_accepted_tokens: [batch_size] number of accepted tokens.
            num_gens: Number of generation requests in the batch.
            num_contexts: Number of context requests in the batch.
            max_draft_len: Number of draft positions to produce.
        """
        self.sa_match_len = torch.zeros((num_gens,), dtype=torch.int32, device="cuda")
        self.sa_draft_tokens = torch.zeros(
            (num_gens, max_draft_len), dtype=torch.int32, device="cuda"
        )
        self.sa_spec_index = 0

        if num_gens > 0:
            gen_request_ids = request_ids[num_contexts:]
            # The CUDA kernel indexes accepted tokens as
            #   acceptedTokensIn[i * (draftLength + 1) + j]
            # so the physical stride must equal max_draft_len + 1.
            # Callers like PARD pad accepted_tokens to [batch, 2K]; the
            # slice + .contiguous() below compacts memory so the stride
            # matches the kernel expectation.
            gen_accepted = accepted_tokens[num_contexts:, : max_draft_len + 1].contiguous()
            match_len, draft_tokens_sa = sa_manager.extend(
                gen_request_ids,
                gen_accepted,
                num_accepted_tokens[num_contexts:],
                max_draft_len,
            )
            self.sa_match_len.copy_(match_len)
            self.sa_draft_tokens.copy_(draft_tokens_sa)

    def maybe_override_all_draft_tokens(
        self,
        draft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Override all K draft positions at once.

        Used by all one-engine workers (MTP, EAGLE3, PARD) to override neural
        draft tokens with SA tokens after the draft loop completes.

        Args:
            draft_tokens: [num_gens, K] draft tokens from the neural drafter.

        Returns:
            The (potentially overridden) draft tokens tensor.
        """
        if self.sa_match_len is not None and self.sa_match_len.shape[0] > 0:
            K = draft_tokens.shape[1]
            mask = (
                (self.sa_match_len >= self.sa_spec_threshold).unsqueeze(1).expand_as(draft_tokens)
            )
            draft_tokens = torch.where(mask, self.sa_draft_tokens[:, :K], draft_tokens)

        return draft_tokens
