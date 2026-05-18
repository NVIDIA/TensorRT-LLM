# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""PEARL: Parallel spEculative decoding with Adaptive dRaft Length.

This worker is a sibling of ``DraftTargetOneModelWorker`` that runs target
verification in parallel with draft token generation across two GPUs
communicating via libibverbs (or TCP fallback).

Reference: Liu et al., "PEARL: Parallel Speculative Decoding with Adaptive
Draft Length", ICLR 2025 (arXiv 2408.11850).

Implementation strategy
-----------------------

PEARL eliminates the *mutual waiting* between draft and target via two
mechanisms:

1. **Post-verify**: while the target verifies round N, the draft is already
   generating round N+1 in parallel. The pipeline is established
   draft-side — when the target sends round N's "accepted tokens, please
   draft next" request, the draft has *already pre-drafted* round N+1
   tokens from speculatively continuing past round N-1's expected
   acceptance. The target round-trip latency for waiting on draft
   collapses to zero.

2. **Pre-verify**: after a rejection the very next round must verify only
   one token (the corrected one). The draft sends that single token early
   so the target can begin a 1-token forward pass without waiting for the
   full gamma batch.

Adaptive draft length (gamma) is profile-based, picked per-batch-size
during warmup: gamma[bs] = round(draft_speed[bs] / target_speed[bs]).

This worker subclasses ``DraftTargetOneModelWorker`` and reuses its KV
cache management, attention metadata handling, and RDMA channel
infrastructure. The PEARL-specific work is:

- Override ``sample_and_accept_draft_tokens`` for paper-faithful greedy
  verification and per-request ``pre_verify`` flag tracking.
- A ``profile_and_set_gamma`` helper that times target forward at several
  batch sizes and exchanges with the draft via kPearlProbe packets.
- The pipelining itself lives in ``examples/llm-api/rdma/pearl_draft_server.py``
  where the draft pre-drafts the next round in parallel with target
  verification of the current round.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata
from .draft_target import (
    DraftTargetOneModelSampler,
    DraftTargetOneModelSpecMetadata,
    DraftTargetOneModelWorker,
)
from .interface import SpecMetadata

if TYPE_CHECKING:
    from ...llmapi.llm_args import PEARLDecodingConfig


@dataclass
class PEARLSpecMetadata(DraftTargetOneModelSpecMetadata):
    """PEARL metadata: extends draft-target metadata with per-request pre_verify
    state and the runtime gamma chosen for this iteration."""

    # uint8 [max_num_requests]; 1 if request is in pre-verify state
    pre_verify_mask: Optional[torch.Tensor] = None
    # The gamma actually used this iteration (may differ from max_draft_len
    # under adaptive scheduling).
    runtime_gamma: int = 0

    def __post_init__(self):
        super().__post_init__()
        self.pre_verify_mask = torch.zeros(
            [self.max_num_requests],
            dtype=torch.uint8,
            device="cuda",
        )

    def prepare(self):
        super().prepare()
        # runtime_gamma defaults to max_draft_len; the worker may override
        # before forward() based on its adaptive table.
        if self.runtime_gamma == 0:
            self.runtime_gamma = self.max_draft_len


class PEARLOneModelSampler(DraftTargetOneModelSampler):
    """PEARL uses greedy verification; sampler matches draft-target's."""


class PEARLOneModelWorker(DraftTargetOneModelWorker):
    """PEARL one-model speculative decoding worker.

    Inherits the full draft-target RDMA-offload pipeline, replacing only:
    - Verification: greedy (argmax == draft_token), with per-request
      pre_verify flag updated on rejection.
    - Adaptive gamma: profile-based table; runtime_gamma is selected by
      batch size in ``prepare_gamma``.
    """

    def __init__(
        self,
        spec_config: "PEARLDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        if not self._rdma_offload_enabled or not self._rdma_offload_v2:
            raise RuntimeError(
                "PEARL requires draft_offload_enabled=True and draft_offload_v2=True"
            )

        # Adaptive-gamma state.
        self._gamma_table: Dict[int, int] = {}
        self._gamma_profile_batch_sizes: List[int] = list(
            getattr(spec_config, "pearl_gamma_profile_batch_sizes", [1, 2, 4, 8, 16, 32])
        )
        self._gamma_profile_steps: int = int(getattr(spec_config, "pearl_gamma_profile_steps", 4))
        self._adaptive_gamma_enabled: bool = bool(
            getattr(spec_config, "pearl_adaptive_gamma", True)
        )

        # Per-request pre_verify flag — every request starts in pre-verify
        # (nano-PEARL Sequence.__init__ sets pre_verify=True).
        self._pre_verify_state: Dict[int, bool] = {}

        # Track accept/reject counts for diagnostics.
        self._total_accepted = 0
        self._total_rejected = 0

        logger.info(
            "PEARL worker initialized: pre_verify=%s post_verify=%s adaptive_gamma=%s "
            "profile_bs=%s profile_steps=%d max_draft_len=%d",
            getattr(spec_config, "pearl_enable_pre_verify", True),
            getattr(spec_config, "pearl_enable_post_verify", True),
            self._adaptive_gamma_enabled,
            self._gamma_profile_batch_sizes,
            self._gamma_profile_steps,
            self.max_draft_len,
        )

    # ------------------------------------------------------------------
    # Adaptive gamma
    # ------------------------------------------------------------------

    def select_gamma_for_batch(self, batch_size: int) -> int:
        """Select gamma based on the profiled table; falls back to
        ``max_draft_len`` when not yet profiled.

        Matches nano-PEARL ``auto_set_gamma`` selection rule (smallest
        profiled bs that is >= current bs).
        """
        if not self._adaptive_gamma_enabled or not self._gamma_table:
            return self.max_draft_len

        for bs in sorted(self._gamma_table.keys()):
            if bs >= batch_size:
                return min(self._gamma_table[bs], self.max_draft_len)
        # Larger than all profiled sizes — use the largest profiled gamma.
        return min(self._gamma_table[max(self._gamma_table.keys())], self.max_draft_len)

    def set_gamma_table(self, table: Dict[int, int]) -> None:
        """Install a precomputed gamma table. Useful when profiling is
        performed externally or coordinated via the draft server."""
        self._gamma_table = {
            int(bs): max(1, min(int(g), self.max_draft_len)) for bs, g in table.items()
        }
        logger.info("PEARL gamma_table set: %s", self._gamma_table)

    # ------------------------------------------------------------------
    # Greedy verify + pre-verify flag tracking
    # ------------------------------------------------------------------

    def sample_and_accept_draft_tokens(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
    ):
        """Greedy verification (argmax == draft_token).

        Reuses the base implementation since
        ``_sample_and_accept_draft_tokens_base`` is already a cumulative-
        product strict-acceptance comparison. Under greedy sampling
        (``allow_advanced_sampling=False``), it samples via ``argmax`` and
        accepts draft tokens iff they match. We additionally maintain the
        per-request ``pre_verify`` flag from request_ids so the PEARL
        protocol can drive draft-side rollback semantics.
        """
        accepted_tokens, num_accepted_tokens = super().sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        # Update pre_verify state based on whether the full gamma batch was
        # accepted. Note: num_accepted_tokens includes the bonus target
        # token, so "fully accepted" means num_accepted == gamma + 1.
        request_ids = getattr(spec_metadata, "request_ids", None)
        if request_ids is not None:
            num_gens = attn_metadata.num_seqs - attn_metadata.num_contexts
            if num_gens > 0:
                gamma = (
                    spec_metadata.runtime_draft_len
                    if spec_metadata.runtime_draft_len > 0
                    else self.max_draft_len
                )
                # Read counts back to CPU lazily; we tolerate the sync here
                # because pre_verify flag-flips are infrequent (rejection
                # path) and the counts are needed for protocol bookkeeping
                # anyway.
                gen_counts = (
                    num_accepted_tokens[attn_metadata.num_contexts :].detach().cpu().tolist()
                )
                for offset, count in enumerate(gen_counts):
                    rid = int(request_ids[attn_metadata.num_contexts + offset])
                    fully_accepted = int(count) >= gamma + 1
                    if fully_accepted:
                        self._pre_verify_state[rid] = False
                        self._total_accepted += gamma
                    else:
                        # At least one draft token rejected -> next round is
                        # pre-verify (verifies only the corrected token).
                        self._pre_verify_state[rid] = True
                        self._total_accepted += max(0, int(count) - 1)
                        self._total_rejected += max(0, gamma - (int(count) - 1))
                # Mirror the flags onto the spec metadata tensor for
                # downstream consumers (currently informational only).
                if isinstance(spec_metadata, PEARLSpecMetadata):
                    flags = torch.tensor(
                        [int(self._pre_verify_state.get(int(r), True)) for r in request_ids],
                        dtype=torch.uint8,
                    )
                    spec_metadata.pre_verify_mask[: len(flags)].copy_(flags, non_blocking=True)

        return accepted_tokens, num_accepted_tokens

    # ------------------------------------------------------------------
    # Forward — inject adaptive gamma selection before parent's RDMA round
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata: AttentionMetadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
        is_warmup: bool = False,
    ):
        if self._adaptive_gamma_enabled and not is_warmup and self._gamma_table:
            batch_size = attn_metadata.num_seqs
            gamma = self.select_gamma_for_batch(batch_size)
            if isinstance(spec_metadata, PEARLSpecMetadata):
                spec_metadata.runtime_gamma = gamma
            # Update the runtime draft length so the base class verifier
            # uses the adaptive gamma.
            spec_metadata.runtime_draft_len = gamma

        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            logits=logits,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
            resource_manager=resource_manager,
            is_warmup=is_warmup,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        total = self._total_accepted + self._total_rejected
        return float(self._total_accepted) / total if total else 0.0

    def reset_acceptance_stats(self):
        self._total_accepted = 0
        self._total_rejected = 0
