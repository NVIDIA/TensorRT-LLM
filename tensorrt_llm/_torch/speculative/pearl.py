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
from ..pyexecutor.sampler import DEFAULT_BEAM_IDX, TorchSampler, add_token
from .draft_target import (
    DraftTargetOneModelSampler,
    DraftTargetOneModelSpecMetadata,
    DraftTargetOneModelWorker,
    _context_prompt_token_batches,
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
    # End token id for each request in the batch.  PEARL uses this before
    # sending accepted state to the external draft server.
    end_ids: Optional[List[int]] = None

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
    """PEARL greedy pipeline sampler.

    Unlike standard speculative decoding, PEARL's target packet returns the
    verified/correction token at an exact position.  When a full gamma window
    matches, the last emitted token is still a draft token, not an extra target
    bonus token.  The generic speculative sampler assumes
    ``new_tokens_lens = accepted_draft_tokens + 1``; PEARL only follows that
    convention on rejection.  On a full match we detect that the final emitted
    token equals the final draft token and keep all gamma draft tokens.
    """

    def update_requests(self, state, resource_manager=None) -> None:
        assert isinstance(state, self.SampleState)

        state.sampler_event.synchronize()
        new_tokens = state.host.new_tokens.tolist()
        new_tokens_lens_list = state.host.new_tokens_lens.tolist()
        next_draft_tokens_list = state.host.next_draft_tokens.tolist()
        beam_idx = DEFAULT_BEAM_IDX
        runtime_draft_len = getattr(state, "runtime_draft_len", self.draft_len)

        for req in state.requests:
            if req.state.name == "GENERATION_COMPLETE":
                continue

            seq_slot = req.py_seq_slot
            num_new_tokens = int(new_tokens_lens_list[seq_slot])
            previous_draft_tokens = list(getattr(req, "py_draft_tokens", []) or [])

            for step in range(num_new_tokens):
                new_token = add_token(req, new_tokens, beam_idx=beam_idx, step=step)
                if TorchSampler._handle_stop_criteria(
                    req, new_token, max_seq_len=self.max_seq_len, beam_idx=beam_idx
                ):
                    break

            accepted_draft_tokens = max(0, num_new_tokens - 1)
            if (
                runtime_draft_len > 0
                and num_new_tokens == runtime_draft_len
                and len(previous_draft_tokens) >= runtime_draft_len
            ):
                last_emitted = int(new_tokens[runtime_draft_len - 1][seq_slot][beam_idx])
                last_draft = int(previous_draft_tokens[runtime_draft_len - 1])
                if last_emitted == last_draft:
                    accepted_draft_tokens = runtime_draft_len

            req.py_num_accepted_draft_tokens = accepted_draft_tokens
            req.py_rewind_len = runtime_draft_len - accepted_draft_tokens
            self._request_common_handling(req, next_draft_tokens_list, runtime_draft_len)


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
    # Early prompt push
    # ------------------------------------------------------------------

    def prepare_target_forward(
        self,
        *,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
        is_warmup: bool = False,
    ) -> None:
        """Push context prompts before the target prefill forward.

        The base draft-target path pushes prompts inside
        ``_rdma_offload_draft_tokens()``, which runs after target logits have
        already been produced. PEARL wants draft prefill to overlap target
        prefill, so the model engine calls this hook as soon as request ids
        and sequence lengths are known, but before target forward.

        ``push_prompt`` binds the logical request to a data-plane route first,
        then sends the TCP prompt under the resulting wire slot. The later
        data-plane packets therefore use the same slot without hard-coding
        request id 0.
        """
        if bool(is_warmup) or self._rdma_v2_offload_layer is None:
            return
        num_contexts = int(getattr(attn_metadata, "num_contexts", 0))
        if num_contexts <= 0 or input_ids is None:
            return
        request_ids = getattr(spec_metadata, "request_ids", None)
        if request_ids is None or len(request_ids) < num_contexts:
            request_ids = getattr(attn_metadata, "request_ids", None)
        if request_ids is None or len(request_ids) < num_contexts:
            return

        context_prompts = _context_prompt_token_batches(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            request_ids=request_ids,
            num_contexts=num_contexts,
        )
        for row in range(num_contexts):
            try:
                self._rdma_v2_offload_layer.push_prompt(
                    int(request_ids[row]),
                    context_prompts[row],
                )
            except RuntimeError as exc:
                logger.warning(
                    "PEARL early prompt push failed for request %s: %s",
                    request_ids[row],
                    exc,
                )
                raise

    # ------------------------------------------------------------------
    # Greedy verify + pre-verify flag tracking
    # ------------------------------------------------------------------

    def sample_and_accept_draft_tokens(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
    ):
        """Greedy exact-match verification for token/position PEARL.

        Target emits the verified/correction token at its exact position.  We
        therefore do not accept the extra target bonus token produced by the
        standard speculative window.  If all gamma draft tokens match, the
        output length is gamma and the last token is the last matched draft
        token.  If token k mismatches, the output length is k + 1 and the last
        token is target's correction at that same position.
        """
        accepted_tokens, num_accepted_tokens = self._sample_and_accept_pearl_tokens(
            logits, attn_metadata, spec_metadata
        )

        # Update pre_verify state based on whether the full gamma batch was
        # accepted. In PEARL token/position mode, full acceptance means the
        # emitted length is gamma and the last emitted token equals the last
        # draft token.
        #
        # Skip this block while CUDA is capturing a graph: the only operations
        # it performs are host<->device syncs (.cpu().item()/.tolist()) used
        # for diagnostic counters and an informational pre_verify mask. Both
        # are never read by the runtime decode path, so dropping them under
        # CUDA-graph mode is correctness-neutral; keeping them would crash
        # capture with cudaErrorStreamCaptureUnsupported.
        request_ids = getattr(spec_metadata, "request_ids", None)
        if request_ids is not None and not torch.cuda.is_current_stream_capturing():
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
                    row = attn_metadata.num_contexts + offset
                    fully_accepted = int(count) >= gamma
                    if fully_accepted and spec_metadata.draft_tokens is not None:
                        draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, gamma)
                        fully_accepted = int(
                            accepted_tokens[row, gamma - 1].detach().cpu().item()
                        ) == int(draft_tokens[offset, gamma - 1].detach().cpu().item())
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

    def _sample_and_accept_pearl_tokens(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: SpecMetadata,
    ):
        batch_size = int(attn_metadata.num_seqs)
        num_contexts = int(attn_metadata.num_contexts)
        num_gens = batch_size - num_contexts
        runtime_draft_len = int(spec_metadata.runtime_draft_len or self.max_draft_len)

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        if spec_metadata.draft_tokens is None:
            draft_tokens = torch.zeros(
                (num_gens, runtime_draft_len),
                dtype=torch.int,
                device=logits.device,
            )
        else:
            draft_tokens = spec_metadata.draft_tokens.reshape(num_gens, runtime_draft_len)

        accepted_tokens = torch.zeros(
            (batch_size, runtime_draft_len + 1),
            dtype=torch.int,
            device=logits.device,
        )
        num_accepted_tokens = torch.ones(
            batch_size,
            dtype=torch.int,
            device=logits.device,
        )

        target_tokens = self._sample_tokens_for_batch(
            logits, spec_metadata, num_contexts, batch_size
        )
        accepted_tokens[:num_contexts, 0] = target_tokens[:num_contexts]

        if num_gens > 0:
            gen_target_tokens = target_tokens[num_contexts:].reshape(
                num_gens, runtime_draft_len + 1
            )
            accepted_tokens[num_contexts:, : runtime_draft_len + 1] = gen_target_tokens

            accepted_draft_counts = torch.cumprod(
                (draft_tokens == gen_target_tokens[:, :runtime_draft_len]).int(),
                dim=-1,
            ).sum(1)
            # PEARL sends the exact verified/correction token.  Full match
            # returns gamma tokens; rejection at index k returns k accepted
            # draft tokens plus one correction token.
            num_accepted_tokens[num_contexts:] = torch.minimum(
                accepted_draft_counts + 1,
                torch.full_like(accepted_draft_counts, runtime_draft_len),
            )

        num_accepted_tokens = self._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, runtime_draft_len
        )
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
