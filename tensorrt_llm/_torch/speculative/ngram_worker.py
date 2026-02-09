# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
NGram speculative decoding worker that runs inside model forward.

This module provides CUDA graph compatible NGram speculative decoding by
integrating the suffix automaton pattern matching into the model's forward pass.

Key components:
- NGramSpecMetadata: Metadata for NGram speculative decoding
- NGramWorker: Spec worker that uses suffix automaton for draft generation
- NGramSampler: Sampler that handles GPU->CPU result extraction
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import torch

from ..pyexecutor.sampler import TorchSampler
from .interface import SpecMetadata, SpecWorkerBase
from .spec_sampler_base import SampleStateSpec, SampleStateTensorsSpec, SpecSamplerBase
from .suffix_automaton import SuffixAutomatonManager

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import NGramDecodingConfig


# Backwards compatibility aliases
SampleStateTensorsNGram = SampleStateTensorsSpec
SampleStateNGram = SampleStateSpec


@dataclass
class NGramSpecMetadata(SpecMetadata):
    """
    Metadata for NGram speculative decoding.

    Holds SA manager reference and GPU buffers for CUDA graph compatibility.
    """

    # Reference to SA manager (state lives outside graph)
    sa_manager: Optional[SuffixAutomatonManager] = None
    # NGram matching configuration
    max_matching_ngram_size: int = -1

    # Pre-allocated GPU buffers for CUDA graph compatibility
    batch_indices_cuda: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_indices_cuda is None and self.max_num_requests > 0:
            self.batch_indices_cuda = torch.zeros(
                self.max_num_requests, dtype=torch.int32, device="cuda"
            )

    def prepare(self) -> None:
        """
        Called BEFORE CUDA graph to set up batch state.

        This method handles:
        1. Setting up batch indices
        2. Preparing SA manager (copies pending states to GPU)

        Note: self.request_ids is set externally before prepare() is called.
        """
        assert self.request_ids is not None, "request_ids must be set before prepare()"
        num_seqs = len(self.request_ids)
        # Set up batch indices
        batch_indices = torch.arange(num_seqs, dtype=torch.int32, device="cpu", pin_memory=True)
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)

        # Prepare SA manager (copies pending states to GPU)
        if self.sa_manager is not None:
            self.sa_manager.prepare(self.request_ids, self.max_draft_len)
        else:
            raise ValueError("SA manager is not set")

    def create_cuda_graph_metadata(self, max_batch_size: int):
        """Creates metadata for CUDA graph execution."""
        if self.is_cuda_graph:
            return self

        import copy

        cuda_graph_metadata = copy.copy(self)
        cuda_graph_metadata.is_cuda_graph = True
        cuda_graph_metadata.max_num_requests = max_batch_size
        cuda_graph_metadata.__post_init__()
        return cuda_graph_metadata


class NGramWorker(SpecWorkerBase):
    """
    NGram speculative decoding worker that runs inside model forward.

    Uses suffix automaton for pattern matching - CUDA graph compatible.
    Unlike MTP which uses neural network layers for drafting, NGram uses
    pattern matching on previously generated tokens.
    """

    def __init__(self, spec_config: "NGramDecodingConfig", model_config=None):
        super().__init__()
        self.spec_config = spec_config
        self._max_draft_len = spec_config.max_draft_len
        self._max_matching_ngram_size = spec_config.max_matching_ngram_size

    @property
    def max_draft_len(self) -> int:
        return self._max_draft_len

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        attn_metadata,
        spec_metadata: NGramSpecMetadata,
        draft_model=None,  # Not used for NGram
    ):
        """
        NGram speculative decoding forward pass.

        Steps:
        1. Sample target logits to get golden token
        2. Accept/reject previous draft tokens
        3. Extend SA with newly accepted tokens
        4. Use extend_ngram to find pattern and generate next drafts
        5. Return accepted_tokens, num_accepted, next_draft_tokens

        Args:
            input_ids: Current input tokens
            position_ids: Position IDs (not used for NGram)
            hidden_states: Hidden states from target model (not used for NGram)
            logits: Target model logits for acceptance
            attn_metadata: Attention metadata
            spec_metadata: NGramSpecMetadata with SA manager
            draft_model: Not used for NGram (pattern matching, not neural)

        Returns:
            Dict with:
            - logits: Raw logits from target model
            - new_tokens: Accepted tokens [batch_size, max_draft_len + 1]
            - new_tokens_lens: Number of accepted tokens [batch_size]
            - next_draft_tokens: Draft tokens for next iteration [batch_size, max_draft_len]
            - next_new_tokens: Input tokens for next iteration [batch_size, max_draft_len + 1]
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        # Step 1-2: Sample and verify draft tokens
        accepted_tokens, num_accepted_tokens = self._sample_and_accept_draft_tokens(
            input_ids, logits, spec_metadata, attn_metadata
        )

        # Step 3-4: Extend SA and generate next draft tokens using GPU kernel
        next_draft_tokens = self._generate_draft_tokens(
            accepted_tokens, num_accepted_tokens, spec_metadata, batch_size, num_contexts
        )

        # Step 5: Prepare next_new_tokens for overlap scheduler
        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }

    def _sample_and_accept_draft_tokens(
        self,
        input_ids: torch.Tensor,
        logits: torch.Tensor,
        spec_metadata: NGramSpecMetadata,
        attn_metadata,
    ):
        """
        Sample and verify draft tokens.

        For context requests: just sample the next token
        For generation requests: verify draft tokens against target tokens
        """
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        # Get draft tokens from spec_metadata (set during prepare)
        draft_tokens = spec_metadata.draft_tokens

        # Initialize to zeros if draft_tokens is None, empty, or has wrong shape
        # spec_metadata.draft_tokens can be:
        # - None (first iteration)
        # - 1D tensor of length num_gens * max_draft_len (from model_engine)
        # - 2D tensor of shape [batch_size, max_draft_len]
        use_zeros = False
        if draft_tokens is None or draft_tokens.numel() == 0:
            use_zeros = True
        elif draft_tokens.dim() == 1:
            # 1D tensor - try to reshape to [num_gens, max_draft_len]
            expected_size = num_gens * self.max_draft_len
            if draft_tokens.numel() == expected_size and num_gens > 0:
                draft_tokens = draft_tokens.reshape(num_gens, self.max_draft_len)
            else:
                use_zeros = True
        elif draft_tokens.dim() == 2:
            # 2D tensor - check shape
            if draft_tokens.shape[-1] != self.max_draft_len:
                use_zeros = True
            else:
                # Slice to get only generation requests' draft tokens
                draft_tokens = draft_tokens[num_contexts:]
        else:
            use_zeros = True

        if use_zeros:
            # No valid draft tokens - create zeros for generation requests
            draft_tokens = torch.zeros(
                (num_gens, self.max_draft_len), dtype=torch.int32, device=logits.device
            )

        # Use base implementation for sampling and acceptance
        accepted_tokens, num_accepted_tokens = self._sample_and_accept_draft_tokens_base(
            logits=logits,
            draft_tokens=draft_tokens,
            num_contexts=num_contexts,
            batch_size=batch_size,
            spec_metadata=spec_metadata,
        )

        return accepted_tokens, num_accepted_tokens

    def _generate_draft_tokens(
        self,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_metadata: NGramSpecMetadata,
        batch_size: int,
        num_contexts: int,
    ) -> torch.Tensor:
        """
        Generate draft tokens using SA extend_ngram - fully on GPU.

        Args:
            accepted_tokens: [batch_size, max_draft_len + 1] accepted tokens
            num_accepted_tokens: [batch_size] number of accepted tokens
            spec_metadata: NGram metadata with SA manager
            batch_size: Total batch size
            num_contexts: Number of context requests

        Returns:
            next_draft_tokens: [batch_size, max_draft_len] draft tokens tensor
        """
        sa_manager = spec_metadata.sa_manager
        request_ids = spec_metadata.request_ids
        max_draft_len = self._max_draft_len

        if sa_manager is None or request_ids is None:
            # No SA manager available, throw error
            raise ValueError("No SA manager available")

        # extend_ngram is CUDA graph compatible
        # It extends SA states with accepted tokens and performs pattern matching
        match_len, draft_tokens = sa_manager.extend_ngram(
            request_ids,
            accepted_tokens,
            num_accepted_tokens,
            max_draft_len,
            max_ngram_size=self._max_matching_ngram_size,
        )

        return draft_tokens  # [batch_size, max_draft_len] GPU tensor


class NGramSampler(SpecSamplerBase):
    """
    Sampler for NGram that extracts GPU results to CPU after graph replay.

    Uses SpecSamplerBase with default behavior (draft_len + 1 storage,
    adds dummy draft tokens for context requests).
    """

    # Use alias for backwards compatibility
    SampleState = SampleStateNGram

    def __init__(self, args: TorchSampler.Args, *, max_draft_len: int):
        super().__init__(args, draft_len=max_draft_len)
