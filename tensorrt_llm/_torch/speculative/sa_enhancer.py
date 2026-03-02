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
        3. Call ``maybe_override_draft_tokens`` inside each draft step (per-layer
           workers like MTP/EAGLE3) or once after all drafts are produced (PARD).
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
            accepted_tokens: [batch_size, max_draft_len + 1] accepted tokens.
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
            match_len, draft_tokens_sa = sa_manager.extend(
                gen_request_ids,
                accepted_tokens[num_contexts:],
                num_accepted_tokens[num_contexts:],
                max_draft_len,
            )
            self.sa_match_len.copy_(match_len)
            self.sa_draft_tokens.copy_(draft_tokens_sa)

    def maybe_override_draft_tokens(
        self,
        draft_tokens: torch.Tensor,
        num_contexts: int,
    ) -> torch.Tensor:
        """Override neural draft tokens with SA draft tokens where match is strong.

        For per-layer workers (MTP, EAGLE3), call this once per draft step —
        it automatically advances ``sa_spec_index``.

        Args:
            draft_tokens: [batch_size] draft tokens from the neural drafter.
            num_contexts: Number of context requests in the batch.

        Returns:
            The (potentially overridden) draft tokens tensor.
        """
        if self.sa_match_len is not None and self.sa_match_len.shape[0] > 0:
            draft_tokens[num_contexts:] = torch.where(
                self.sa_match_len >= self.sa_spec_threshold,
                self.sa_draft_tokens[:, self.sa_spec_index],
                draft_tokens[num_contexts:],
            )
            self.sa_spec_index += 1

        return draft_tokens
