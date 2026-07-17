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
#
# DSpark worker / metadata mirror the DFlash plumbing (capture target-layer
# hidden states, accept the previous block with standard verification, draft a
# new block in one backbone forward), adapted to DSpark's draft model which
# produces the whole block (and its confidence-truncated length) inside a single
# ``DSparkDraftModel.forward`` rather than via mask-token cross-attention.

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .interface import SpecMetadata, SpecWorkerBase

if TYPE_CHECKING:
    from ...llmapi.llm_args import DSparkDecodingConfig


@dataclass
class DSparkSpecMetadata(SpecMetadata):
    """Metadata for DSpark speculative decoding.

    Captures hidden states from the target model's ``layers_to_capture`` during
    the target forward pass. DSpark captures the *mean over the multi-head
    (mHC) residual streams* at each captured layer (handled by the target-side
    capture hook), concatenated across layers, and feeds them to the draft
    model's ``main_proj`` + ``main_norm`` (inside ``DSparkDraftModel.forward``)
    as the captured-context attention input (``main_x``).

    Mirrors :class:`DFlashSpecMetadata`; the only DSpark-specific detail is that
    the per-layer captured width is the model hidden size (post hc-mean), so the
    buffer is ``[max_num_tokens, hidden_size * num_capture_layers]``.
    """

    batch_indices_cuda: Optional[torch.Tensor] = None

    # Hidden state capture fields
    layers_to_capture: Optional[List[int]] = None
    hidden_size: int = 0
    max_num_tokens: int = 0
    dtype: torch.dtype = torch.bfloat16
    captured_hidden_states: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.batch_indices_cuda = torch.empty(
            [self.max_num_requests],
            dtype=torch.int,
            device="cuda",
        )

        self.is_spec_dec_tree = False
        self.is_spec_dec_dynamic_tree = False

        # Set up hidden state capture buffer
        if self.layers_to_capture is not None and len(self.layers_to_capture) > 0:
            self.layers_to_capture = sorted(list(self.layers_to_capture))
            self.num_capture_layers = len(self.layers_to_capture)
            # O(1) lookups for is_layer_capture() and maybe_capture_hidden_states()
            self._capture_layer_set = frozenset(self.layers_to_capture)
            self._layer_to_idx = {lid: i for i, lid in enumerate(self.layers_to_capture)}
            self.captured_hidden_states = torch.empty(
                (self.max_num_tokens, self.hidden_size * self.num_capture_layers),
                dtype=self.dtype,
                device="cuda",
            )
            logger.info(
                f"DSpark: capturing hidden states from layers {self.layers_to_capture}, "
                f"buffer shape {self.captured_hidden_states.shape}"
            )
        else:
            self.num_capture_layers = 0
            self._capture_layer_set = frozenset()
            self._layer_to_idx = {}

    def prepare(self):
        assert self.request_ids is not None
        num_seqs = len(self.request_ids)
        batch_indices = torch.arange(
            num_seqs, dtype=torch.int, device="cpu", pin_memory=prefer_pinned()
        )
        self.batch_indices_cuda[:num_seqs].copy_(batch_indices, non_blocking=True)

        # CUDA-graph-safe path: maintain the request->slot mapping on the host
        # (outside the captured region) and mirror it into ``_batch_to_slot`` so the
        # captured gen forward can index the rolling windows by tensor. Mirrors
        # ``DFlashSpecMetadata.prepare`` (dflash.py:96-113).
        worker = getattr(self, "_dspark_worker", None)
        if worker is not None and worker._win_inited:
            current = set(self.request_ids)
            for rid in list(worker._req_to_slot.keys()):
                if rid not in current:
                    slot = worker._req_to_slot.pop(rid)
                    worker._ctx_len[slot] = 0
                    worker._kv_windows[slot].zero_()
                    worker._free_slots.append(slot)
            # Unknown request IDs (e.g. synthetic warmup requests) default to slot 0.
            mapping = torch.tensor(
                [worker._req_to_slot.get(rid, 0) for rid in self.request_ids],
                dtype=torch.long,
                device="cpu",
                pin_memory=prefer_pinned(),
            )
            worker._batch_to_slot[:num_seqs].copy_(mapping, non_blocking=True)

    def is_layer_capture(self, layer_id: int) -> bool:
        return layer_id in self._capture_layer_set

    def maybe_capture_hidden_states(
        self, layer_id: int, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> None:
        """Capture hidden states from a target model layer into the buffer.

        DeepSeek-V4 keeps the multi-head (mHC) residual stream flattened as
        ``[num_tokens, hc_mult * hidden]``; DSpark captures the *mean over the hc
        streams* (reference ``h.mean(dim=2)`` with ``h`` shaped
        ``[*, hc_mult, hidden]``). We reduce here so the V4 decoder layer's
        existing capture call is unchanged. A ``[num_tokens, hidden]`` input
        (already reduced / non-mHC) is stored as-is.
        """
        if self.captured_hidden_states is None:
            return
        i = self._layer_to_idx.get(layer_id)
        if i is not None:
            num_tokens = hidden_states.shape[0]
            to_save = hidden_states + residual if residual is not None else hidden_states
            # mHC residual -> mean over the hc_mult streams.
            if to_save.shape[-1] != self.hidden_size:
                hc_mult = to_save.shape[-1] // self.hidden_size
                to_save = to_save.reshape(num_tokens, hc_mult, self.hidden_size).mean(dim=1)
            self.captured_hidden_states[
                :num_tokens, i * self.hidden_size : (i + 1) * self.hidden_size
            ].copy_(to_save, non_blocking=True)

    def get_hidden_states(self, num_tokens: int) -> Optional[torch.Tensor]:
        """Get captured hidden states (all layers concatenated)."""
        if self.captured_hidden_states is None:
            return None
        return self.captured_hidden_states[
            :num_tokens, : self.hidden_size * self.num_capture_layers
        ]


class DSparkWorker(SpecWorkerBase):
    """Worker for DSpark speculative decoding.

    DSpark drafts a whole block of ``block_size`` tokens in one backbone forward
    (``DSparkDraftModel.forward``): it projects the captured target-layer hidden
    states (``main_proj`` + ``main_norm``) into the draft's captured-context
    attention, runs the ``num_stages`` DSpark blocks over a rolling captured
    window, refines the per-position logits with the Markov head, and predicts a
    per-position acceptance confidence used to truncate the proposed prefix.

    Unlike DFlash, the draft does NOT use the paged KV cache or mask-token
    cross-attention: its attention K/V come from the worker-owned rolling window
    of projected captured context (one ``main_kv`` per decode step, per stage).
    Acceptance of the previous block goes through the unified
    :meth:`SpecWorkerBase.sample_and_accept_draft_tokens` (strict target-verify,
    or rejection sampling for a non-greedy batch), so greedy parity with no-spec
    is preserved regardless of draft quality.

    The rolling window is kept consistent across the whole decode: it is seeded
    from the prompt's captured context at prefill and back-filled with the
    intermediate accepted tokens of a multi-accept step (both via
    ``DSparkDraftModel.write_context_windows``), in addition to the per-step bonus
    write done by the generation path. These affect draft acceptance rate only,
    not correctness, which the standard target verify guarantees.

    Reference: DeepSeek DeepSpec (https://github.com/deepseek-ai/DeepSpec).
    """

    def __init__(
        self,
        spec_config: "DSparkDecodingConfig",
        mapping: Mapping,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(use_separate_draft_kv_cache)
        self.spec_config = spec_config
        self.mapping = mapping

        # Per-slot rolling captured-context KV windows, built lazily on the
        # first forward (fixed-size for slot-indexed reads/writes).
        self._win_inited = False
        self._kv_windows: Optional[torch.Tensor] = None  # [max_batch, num_stages, win, hd]
        self._ctx_len: Optional[torch.Tensor] = None  # [max_batch] abs decode position
        self._win = 0

        # Slot management. ``_req_to_slot`` (python dict) + ``_free_slots`` are the
        # source of truth, updated in prepare()/forward(); ``_batch_to_slot`` is the
        # CUDA mirror (request-order -> slot) read by the CUDA-graph-safe batched
        # gen path (set on the host in prepare(), so the captured forward indexes
        # the rolling windows through a tensor instead of a python dict lookup).
        self._req_to_slot = {}  # request_id -> slot index
        self._free_slots = deque()  # available slot indices
        self._batch_to_slot: Optional[torch.Tensor] = None  # [max_batch] long, cuda

        # The generation draft path is the batched, host-sync-free
        # ``_draft_gen_block_batched`` + ``DSparkDraftModel.forward_batched`` +
        # ``dspark_attention_forward_batched``: it is correct in eager mode AND safe
        # to capture into the target's CUDA graph (DSpark is a one-engine drafter —
        # its worker forward runs inside that graph, so the draft path MUST be
        # capture-safe whenever ``cuda_graph_config`` is set).

        logger.info(
            f"DSparkWorker initialized with "
            f"use_separate_draft_kv_cache={use_separate_draft_kv_cache}"
        )

    @property
    def max_draft_len(self) -> int:
        return self.spec_config.max_draft_len

    def _lazy_init(self, draft_model, spec_metadata) -> None:
        block_size = int(draft_model.block_size)
        if block_size != self.max_draft_len:
            raise ValueError(
                "DSpark draft model block_size must equal worker max_draft_len; "
                f"got block_size={block_size} and max_draft_len={self.max_draft_len}"
            )

        if self._win_inited:
            return
        max_batch = spec_metadata.max_num_requests
        num_stages = draft_model.num_stages
        self._win = int(draft_model._attn_params["window_size"])
        head_dim = int(draft_model._attn_params["head_dim"])

        self._kv_windows = torch.zeros(
            (max_batch, num_stages, self._win, head_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        self._ctx_len = torch.zeros(max_batch, dtype=torch.long, device="cuda")
        self._batch_to_slot = torch.zeros(max_batch, dtype=torch.long, device="cuda")
        self._free_slots = deque(range(max_batch))
        self._req_to_slot = {}
        self._win_inited = True
        logger.info(
            f"DSpark: allocated rolling KV windows "
            f"[{max_batch}, {num_stages}, {self._win}, {head_dim}]"
        )

    def _assign_slot(self, req_id: int, reset: bool) -> int:
        """Get (or refresh) the slot for a request; reset clears its window."""
        if reset and req_id in self._req_to_slot:
            old = self._req_to_slot.pop(req_id)
            self._ctx_len[old] = 0
            self._kv_windows[old].zero_()
            self._free_slots.append(old)
        if req_id not in self._req_to_slot:
            if not self._free_slots:
                raise RuntimeError(
                    "DSpark has no free rolling-window slots for request "
                    f"{req_id}; increase max_num_requests"
                )
            slot = self._free_slots.popleft()
            self._req_to_slot[req_id] = slot
            self._ctx_len[slot] = 0
            self._kv_windows[slot].zero_()
        return self._req_to_slot[req_id]

    def _seed_context_windows(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ) -> None:
        """Seed context chunks using their absolute positions.

        A request can arrive in multiple prefill chunks. Only its first chunk
        starts at position zero and resets the persistent rolling window;
        continuation chunks append to the same request slot.
        """
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        flat_position_ids = position_ids.reshape(-1)
        context_offset = 0
        for i in range(attn_metadata.num_contexts):
            chunk_len = int(attn_metadata._seq_lens[i])
            chunk_positions = flat_position_ids[context_offset : context_offset + chunk_len].long()
            if chunk_len == 0:
                context_offset += chunk_len
                continue

            req_id = spec_metadata.request_ids[i]
            first_position = int(chunk_positions[0].item())
            slot = self._assign_slot(req_id, reset=first_position == 0)
            self._ctx_len[slot] = chunk_positions[-1] + 1

            if captured is not None:
                keep = min(self._win, chunk_len)
                hidden = captured[context_offset + chunk_len - keep : context_offset + chunk_len]
                # A prompt token at absolute position p is stored in frame p+1,
                # matching the generation path's start_pos convention.
                window_positions = chunk_positions[-keep:] + 1
                draft_model.write_context_windows(hidden, window_positions, self._kv_windows[slot])
            context_offset += chunk_len

    def _draft_gen_block_batched(
        self,
        draft_model,
        spec_metadata: "DSparkSpecMetadata",
        attn_metadata,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
        total_target_tokens: int,
        all_rank_num_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """CUDA-graph-safe batched gen draft (all gen requests in one forward).

        Free of host syncs and data-dependent shapes: per-request quantities
        (``nacc``, the bonus, ``main_hidden``, ``start_pos``, the multi-accept
        back-fill) are gathered as tensors, slots come from the host-built
        ``_batch_to_slot`` mirror, and the backbone runs once via
        ``DSparkDraftModel.forward_batched``. Returns the per-position corrected
        block logits ``[num_gens, K, vocab]`` (or ``None`` when there is nothing to
        draft); the worker feeds them to ``SpecWorkerBase.sample_draft_tokens``.
        Confidence truncation stays disabled — the full block is proposed.
        """
        num_gens = batch_size - num_contexts
        K = self.max_draft_len
        Kp1 = K + 1
        device = accepted_tokens.device

        if num_gens == 0:
            return None
        captured = spec_metadata.get_hidden_states(total_target_tokens)
        if captured is None:
            return None

        # gen-only graph batches have num_ctx_tokens == 0; mixed eager batches put
        # the gen tokens after the context tokens.
        gen_start = attn_metadata.num_ctx_tokens
        slots = self._batch_to_slot[num_contexts:batch_size]  # [G]
        nacc = num_accepted_tokens[num_contexts:batch_size].long()  # [G]
        gidx = nacc - 1  # [G] index of the bonus within each verified prefix

        # Bonus token = last accepted token of the verified prefix.
        bonus = (
            accepted_tokens[num_contexts:batch_size].gather(1, gidx.unsqueeze(1)).squeeze(1).long()
        )  # [G]

        # Captured target hidden at the bonus position within each request's Kp1
        # processed tokens.
        arange_g = torch.arange(num_gens, device=device)
        base = gen_start + arange_g * Kp1  # [G]
        main_hidden = captured[base + gidx]  # [G, ncap*hidden]

        # Fixed-size ([G, K]) masked back-fill of the intermediate accepted tokens
        # (everything but the bonus) into the rolling window — same frames as the
        # eager path (old+1 .. old+nacc-1), with j >= nacc-1 masked out.
        old = self._ctx_len[slots]  # [G] pre-increment decode position
        j = torch.arange(K, device=device)  # [K]
        interim_valid = j.unsqueeze(0) < (nacc.unsqueeze(1) - 1)  # [G, K]
        interim_pos = old.unsqueeze(1) + 1 + j.unsqueeze(0)  # [G, K]
        interim_base = (base.unsqueeze(1) + j.unsqueeze(0)).clamp(
            min=0, max=captured.shape[0] - 1
        )  # [G, K] (clamped; invalid entries are masked out anyway)
        interim_hidden = captured[interim_base]  # [G, K, ncap*hidden]
        draft_model.write_context_windows_batched(
            interim_hidden, interim_pos, slots, interim_valid, self._kv_windows
        )

        # Advance the decode position by the accepted count; start_pos (= post-
        # increment ctx_len) matches the eager path's frame value.
        start_pos = old + nacc  # [G]
        self._ctx_len[slots] = start_pos

        # Surface the per-position corrected block logits ([num_gens, K, vocab])
        # and let SpecWorkerBase.sample_draft_tokens do the (greedy or rejection)
        # sampling + TP gather + draft_probs scatter, rather than argmaxing here.
        _toks, _num_proposed, block_logits = draft_model.forward_batched(
            main_hidden,
            bonus,
            start_pos,
            kv_windows=self._kv_windows,
            slots=slots,
            temperature=0.0,
            confidence_threshold=0.0,
            return_logits=True,
            all_rank_num_tokens=all_rank_num_tokens,
        )
        return block_logits

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        logits,
        attn_metadata,
        spec_metadata,
        draft_model,
        resource_manager=None,
    ):
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        raw_logits = logits
        K = self.max_draft_len

        self._lazy_init(draft_model, spec_metadata)
        # Backref so DSparkSpecMetadata.prepare() can maintain the host slot map
        # and mirror it into _batch_to_slot for the CUDA-graph-safe gen path.
        spec_metadata._dspark_worker = self
        self._execute_guided_decoder_if_present(logits)

        # Target-verify acceptance via the unified SpecWorkerBase entry: it
        # reshapes the stored draft tokens (default (num_gens, runtime_draft_len)
        # hook), then routes to strict or rejection sampling. Greedy parity with
        # the previous hand-rolled path is preserved (rejection only engages for a
        # non-greedy batch with valid draft_probs).
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            logits, attn_metadata, spec_metadata
        )

        total_target_tokens = input_ids.shape[0]

        # CUDA-graph warmup guard: the warmup forwards (is_cuda_graph set, stream
        # NOT yet capturing) run synthetic gen batches that would otherwise advance
        # the persistent rolling-window state. Snapshot and restore it so warmup is
        # side-effect-free. (During the capture pass itself the stream IS capturing,
        # so we skip the save/restore and let the ops be recorded; real requests
        # reset their slot's window+ctx_len at prefill, wiping any capture-time
        # mutation.)
        is_warmup = (
            getattr(spec_metadata, "is_cuda_graph", False)
            and not torch.cuda.is_current_stream_capturing()
        )
        if is_warmup:
            saved_ctx_len = self._ctx_len.clone()
            saved_windows = self._kv_windows.clone()

        # Assign / reset window slots for context (prefill) requests and seed each
        # request's rolling KV window from its prompt's captured context, so the
        # first generation step drafts against real context instead of an all-zero
        # window (acceptance-rate only; verified decoding keeps output correct).
        if num_contexts > 0:
            self._seed_context_windows(
                draft_model,
                spec_metadata,
                attn_metadata,
                position_ids,
                total_target_tokens,
            )

        # FUSED_COMM MoE backends (DeepGEMM MegaMoE) synchronize EP ranks with an
        # in-kernel phase-flip NVLink barrier that flips on every kernel call, so
        # every rank must invoke the draft MoE the same number of times and with
        # the same globally-gathered per-rank token list, or the barrier desyncs
        # (hang / "unspecified launch failure"). The draft runs over generation
        # requests only, each expanded to ``block`` positions, so the per-rank
        # draft-MoE token count is ``num_gens * block``. ``all_rank_num_gens`` is
        # gathered at metadata-prep time (model_engine, outside any CUDA-graph
        # capture region); it is None for non-ADP / single-rank runs, where the
        # local ``[num_tokens]`` fallback in ``_forward_stage`` is correct.
        block = int(draft_model.block_size)
        all_rank_num_gens = getattr(spec_metadata, "all_rank_num_gens", None)
        # A rank with zero local gen requests still has to cross the draft MoE's
        # cross-rank barrier, but DeepseekV4MoE's router / shared-expert dense
        # GEMMs reject a 0-row input (cuBLAS CUBLAS_STATUS_INVALID_VALUE), so such
        # a rank runs a single 1-row dummy through the MoE (like ADP padding).
        # Encode that as ``1`` in the globally-shared per-rank token list so every
        # rank agrees on the FUSED_COMM chunk count and per-rank slice.
        all_rank_draft_tokens = (
            [max(1, int(g) * block) for g in all_rank_num_gens]
            if all_rank_num_gens is not None
            else None
        )
        global_has_gen = (
            max(all_rank_num_gens) > 0 if all_rank_num_gens is not None else num_gens > 0
        )

        if num_gens > 0:
            # The batched gen-block draft returns the per-position corrected block
            # logits [num_gens, K, vocab] and is CUDA-graph-safe.
            gen_logits = self._draft_gen_block_batched(
                draft_model,
                spec_metadata,
                attn_metadata,
                accepted_tokens,
                num_accepted_tokens,
                num_contexts,
                batch_size,
                total_target_tokens,
                all_rank_num_tokens=all_rank_draft_tokens,
            )
            if gen_logits is not None:
                # SpecWorkerBase samples the draft tokens (greedy argmax, or
                # rejection sampling for a non-greedy batch), performs the TP
                # gather, and scatters the proposal distribution into draft_probs.
                gen_draft_tokens = self.sample_draft_tokens(
                    gen_logits, spec_metadata, batch_size, num_contexts=num_contexts
                )
                # The context one-hot must match the width the gen scatter just
                # published to draft_probs, NOT gen_logits.shape[-1]: under TP the
                # draft logits are vocab-sharded and sample_draft_tokens gathers
                # them to full vocab before scattering, so the pre-gather shard
                # width would leave stale columns and corrupt rejection.
                gen_vocab = spec_metadata.draft_probs_last_dim
            else:
                gen_draft_tokens = torch.zeros((num_gens, K), dtype=torch.int32, device="cuda")
                gen_vocab = None
        else:
            # No local generation requests: if any peer EP rank has some, we must
            # still cross the draft MoE's cross-rank barrier the same number of
            # times (zero-token) so a FUSED_COMM phase-flip barrier stays lockstep.
            if global_has_gen:
                draft_model.run_moe_lockstep_noop(all_rank_draft_tokens, accepted_tokens.device)
            gen_draft_tokens = torch.empty((0, K), dtype=torch.int32, device="cuda")
            gen_vocab = None

        # Context requests are not drafted by the block worker (zero placeholder
        # token); fill their draft-prob slot rows with a legal one-hot so they are
        # a valid distribution when they become gen requests next iteration.
        self.write_context_onehot_draft_probs(spec_metadata, num_contexts, num_gens, K, gen_vocab)

        if num_contexts > 0:
            ctx_draft_tokens = torch.zeros((num_contexts, K), dtype=torch.int32, device="cuda")
            next_draft_tokens = torch.cat([ctx_draft_tokens, gen_draft_tokens], dim=0)
        else:
            next_draft_tokens = gen_draft_tokens

        next_new_tokens = self._prepare_next_new_tokens(
            accepted_tokens,
            next_draft_tokens,
            spec_metadata.batch_indices_cuda,
            batch_size,
            num_accepted_tokens,
        )

        if is_warmup:
            self._ctx_len.copy_(saved_ctx_len)
            self._kv_windows.copy_(saved_windows)

        return {
            "logits": raw_logits,
            "new_tokens": accepted_tokens,
            "new_tokens_lens": num_accepted_tokens,
            "next_draft_tokens": next_draft_tokens,
            "next_new_tokens": next_new_tokens,
        }
