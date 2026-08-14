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

    The SA extend+search kernels are launched on a dedicated CUDA side-stream
    so they overlap with the compute-heavy draft model forward passes on the
    main stream.  Results are synchronized lazily — only when
    ``maybe_override_all_draft_tokens`` is called after the draft loop.

    Usage:
        1. Construct once during worker ``__init__`` when ``sa_config`` is set.
        2. Call ``extend_and_prepare`` after ``sample_and_accept_draft_tokens``.
        3. Call ``maybe_override_all_draft_tokens`` once after all draft layers
           have finished, so that neural draft layers never see SA tokens.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold
        self.sa_match_len: Optional[torch.Tensor] = None
        self.sa_draft_tokens: Optional[torch.Tensor] = None
        self.sa_spec_index: int = 0
        self._sa_stream: Optional[torch.cuda.Stream] = None
        self._sa_event: Optional[torch.cuda.Event] = None
        self._num_gens: int = 0

    def _ensure_stream(self) -> None:
        if self._sa_stream is None:
            self._sa_stream = torch.cuda.Stream()
            self._sa_event = torch.cuda.Event()

    def _ensure_buffers(self, num_gens: int, max_draft_len: int) -> None:
        """Pre-allocate / reuse GPU buffers for SA match results."""
        if self.sa_match_len is None or self.sa_match_len.shape[0] < num_gens:
            self.sa_match_len = torch.zeros((num_gens,), dtype=torch.int32, device="cuda")
        if (
            self.sa_draft_tokens is None
            or self.sa_draft_tokens.shape[0] < num_gens
            or self.sa_draft_tokens.shape[1] < max_draft_len
        ):
            self.sa_draft_tokens = torch.zeros(
                (num_gens, max_draft_len), dtype=torch.int32, device="cuda"
            )

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
        draft generation loop.  The SA kernels are launched on a side-stream so
        the caller can immediately proceed with the draft model forward passes.

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
        self._ensure_buffers(num_gens, max_draft_len)
        self._num_gens = num_gens
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

            self._ensure_stream()
            main_stream = torch.cuda.current_stream()
            self._sa_stream.wait_stream(main_stream)

            with torch.cuda.stream(self._sa_stream):
                if sa_manager.enable_global_pool:
                    match_len, draft_tokens_sa = sa_manager.extend_global(
                        gen_request_ids,
                        gen_accepted,
                        num_accepted_tokens[num_contexts:],
                        max_draft_len,
                    )
                else:
                    match_len, draft_tokens_sa = sa_manager.extend(
                        gen_request_ids,
                        gen_accepted,
                        num_accepted_tokens[num_contexts:],
                        max_draft_len,
                    )
                self.sa_match_len[:num_gens].copy_(match_len)
                self.sa_draft_tokens[:num_gens, :max_draft_len].copy_(draft_tokens_sa)

            self._sa_event.record(self._sa_stream)

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
        if self.sa_match_len is not None and self._num_gens > 0:
            if self._sa_event is not None:
                torch.cuda.current_stream().wait_event(self._sa_event)

            n = self._num_gens
            K = draft_tokens.shape[1]
            mask = (self.sa_match_len[:n] >= self.threshold).unsqueeze(1).expand_as(draft_tokens)
            draft_tokens = torch.where(mask, self.sa_draft_tokens[:n, :K], draft_tokens)

        return draft_tokens
