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
Eagle3 one-model dynamic tree speculative decoding.

This module separates the dynamic tree logic from the base Eagle3 one-model
worker (eagle3.py) for clearer architecture. The base eagle3.py handles linear
tree mode and shared infrastructure; this file handles:

- Eagle3OneModelDynamicTreeWorker: draft loop, verification, tree construction
- Eagle3OneModelDynamicTreeSampler: sampler with accepted indices tracking
- Buffer management for dynamic tree operations (history, scores, parents, masks)

Goal: CUDA graph accept rate must match eager accept rate exactly.
"""

from typing import TYPE_CHECKING, override

import torch

from ..attention_backend import AttentionMetadata
from ..pyexecutor.llm_request import LlmRequestState
from ..pyexecutor.sampler import TorchSampler
from .eagle3 import Eagle3OneModelWorker
from .mtp import MTPSampler
from .spec_tree_manager import SpecTreeManager

if TYPE_CHECKING:
    from ...llmapi.llm_args import EagleDecodingConfig


class Eagle3OneModelDynamicTreeSampler(MTPSampler):
    """Sampler for one-model EAGLE3 dynamic tree mode.

    Extends MTPSampler with accepted draft token indices tracking, which is
    needed for KV cache rewind after dynamic tree verification.
    """

    def __init__(self, args: TorchSampler.Args, spec_config=None):
        super().__init__(args, nextn=args.max_total_draft_tokens)
        seq_slots = args.max_num_sequences
        max_draft_len = spec_config.max_draft_len

        self._accepted_indices_store = torch.full(
            (seq_slots, max_draft_len), -1, dtype=torch.int32, device="cuda"
        )

    def sample_async(self, scheduled_requests, outputs, num_context_logits_prefix_sum):
        if "accepted_draft_tokens_indices" in outputs:
            requests = scheduled_requests.all_requests()
            slots = torch.as_tensor([r.py_seq_slot for r in requests], device="cuda")
            indices = outputs["accepted_draft_tokens_indices"][: len(requests)]
            self._accepted_indices_store.index_copy_(0, slots, indices)
        return super().sample_async(scheduled_requests, outputs, num_context_logits_prefix_sum)

    def update_requests(self, state, resource_manager=None):
        super().update_requests(state, resource_manager)
        for req in state.scheduled_requests.generation_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            n_accepted = req.py_num_accepted_draft_tokens
            if n_accepted > 0:
                slot = req.py_seq_slot
                req.py_num_accepted_draft_tokens_indices = (
                    self._accepted_indices_store[slot, :n_accepted].cpu().tolist()
                )


class Eagle3OneModelDynamicTreeWorker(Eagle3OneModelWorker):
    """Eagle3 one-model worker with dynamic tree support.

    Inherits linear tree functionality from Eagle3OneModelWorker and adds
    dynamic tree draft loop, verification, and tree construction.
    """

    def __init__(
        self, spec_config: "EagleDecodingConfig", mapping, use_separate_draft_kv_cache: bool = False
    ):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        assert self.use_dynamic_tree, (
            "Eagle3OneModelDynamicTreeWorker requires use_dynamic_tree=True"
        )

        from .dynamic_tree_ops import DynamicTreeOpsConverter

        K = spec_config.dynamic_tree_max_topK
        max_draft_len = spec_config.max_draft_len
        max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
        # Read runtime max_batch_size set by py_executor_creator; fall back to 256
        max_batch_size = getattr(spec_config, "_runtime_max_batch_size", 256)
        self._max_batch_size = max_batch_size

        self.K = K
        self.max_total_draft_tokens = max_total_draft_tokens
        self.tokens_per_gen_step = max_total_draft_tokens + 1

        # 2D buffers: [max_batch, max_total] avoids per-layer cat
        self.draft_tokens_buffer = torch.zeros(
            max_batch_size, max_total_draft_tokens, dtype=torch.int64, device="cuda"
        )
        tokens_per_gen_step = self.tokens_per_gen_step
        self.position_ids_buffer = torch.zeros(
            max_batch_size, tokens_per_gen_step, dtype=torch.int64, device="cuda"
        )
        self.history_draft_tokens_buffer = torch.zeros(
            (max_batch_size, (K + K * K * (max_draft_len - 1))), dtype=torch.int64, device="cuda"
        )
        self.history_score_buffer = torch.zeros(
            (max_batch_size, K + K * K * (max_draft_len - 1)), dtype=torch.float32, device="cuda"
        )
        self.history_draft_tokens_parent_buffer = torch.zeros(
            (max_batch_size, K * (max_draft_len - 1) + 1), dtype=torch.int64, device="cuda"
        )
        self.tree_mask_buffer = torch.zeros(
            (max_batch_size * tokens_per_gen_step * tokens_per_gen_step),
            dtype=torch.int32,
            device="cuda",
        )
        self.tree_mask_init_buffer = (
            torch.eye(K, dtype=torch.int32, device="cuda").unsqueeze(0).repeat(max_batch_size, 1, 1)
        )
        self.tree_ops_converter = DynamicTreeOpsConverter(
            dynamic_tree_max_topK=K,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=max_total_draft_tokens,
            max_batch_size=max_batch_size,
            device=torch.device("cuda"),
        )

        # Pre-allocated buffer for accepted draft indices (reused each call)
        self._accepted_draft_indices_tensor = torch.full(
            (max_batch_size, max_draft_len), -1, dtype=torch.int32, device="cuda"
        )

        # Initialized by _sample_and_accept_dynamic_tree; None during warmup
        # when linear fallback verification is used.
        self._accept_token = None
        self._last_num_accepted = None
        self._last_selected_parents = None

        # Hidden state management buffers (initialized in update_hidden_states step 0)
        self._hs_write_buffer = None
        self._hs_read_map = torch.zeros(
            max_batch_size, max_draft_len * K, dtype=torch.long, device="cuda"
        )
        self._accumulated_hs = None
        self._step0_hs = None
        self._hs_dim = None

        # Step 0 reusable buffers (avoid per-call allocation)
        max_step0_tokens = max_draft_len + 1
        self._step0_hs_buf = None  # lazily init (hidden_dim unknown at init)
        self._step0_causal_mask = torch.tensor(
            [(1 << (t + 1)) - 1 for t in range(max_step0_tokens)],
            dtype=torch.int64,
            device="cuda",
        )
        # Pre-allocated index tensors (avoid per-call torch.arange GPU allocations)
        self._arange_max_batch = torch.arange(max_batch_size, device="cuda")
        self._arange_max_path = torch.arange(max_step0_tokens, device="cuda")
        self._causal_offs = torch.arange(max_step0_tokens, device="cuda", dtype=torch.int32)
        self._step0_path_pos_buf = torch.zeros(
            max_batch_size, max_step0_tokens, dtype=torch.long, device="cuda"
        )
        self._step0_gather_ids = torch.empty(0, dtype=torch.long, device="cuda")
        self._parent_init_arange = torch.arange(-1, K, device="cuda", dtype=torch.int32)

        # Pre-allocated buffers for _sample_and_accept_dynamic_tree
        N = tokens_per_gen_step  # includes root
        max_path_len = max_draft_len + 1
        self._accepted_tokens_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int32, device="cuda"
        )
        self._num_accepted_tokens_buf = torch.ones(max_batch_size, dtype=torch.int32, device="cuda")
        self._target_tokens_buf = torch.zeros(max_batch_size * N, dtype=torch.int64, device="cuda")
        self._candidates_buf = torch.zeros(max_batch_size, N, dtype=torch.int64, device="cuda")
        self._target_predict_buf = torch.zeros(max_batch_size, N, dtype=torch.int64, device="cuda")

        # Pre-allocated buffers for prepare_1st_drafter_inputs torch.cat replacements
        max_total_tokens = max_batch_size * N  # conservative upper bound
        self._step0_input_ids_buf = torch.zeros(max_total_tokens, dtype=torch.int64, device="cuda")
        self._step0_position_ids_buf = torch.zeros(
            max_total_tokens, dtype=torch.int64, device="cuda"
        )
        # hidden_states buf lazily initialized (hidden_dim unknown at init)
        self._step0_hidden_states_buf = None

        # Pre-allocated buffer for gather_ids cat in _forward_draft_loop
        self._gather_ids_buf = torch.zeros(max_total_tokens, dtype=torch.long, device="cuda")

        # Pre-allocated buffers for prepare_tree_mask_and_position_offset
        self._current_mask_buf = torch.zeros(
            max_batch_size,
            max_total_draft_tokens,
            tokens_per_gen_step,
            dtype=torch.int32,
            device="cuda",
        )
        self._new_pos_offset_buf = torch.zeros(
            max_batch_size, max_total_draft_tokens, dtype=torch.int32, device="cuda"
        )

    def _ensure_spec_tree_manager(self, resource_manager):
        """Lazily initialize spec_tree_manager from resource_manager."""
        if self.spec_tree_manager is None and resource_manager is not None:
            from ..pyexecutor.resource_manager import ResourceManagerType

            spec_rm = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER
            )
            if spec_rm is not None and hasattr(spec_rm, "spec_tree_manager"):
                self.spec_tree_manager = spec_rm.spec_tree_manager

    # ---- Overridden dispatch methods ----

    @override
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
        """Override to add accepted_draft_tokens_indices to output."""
        output = super().forward(
            input_ids,
            position_ids,
            hidden_states,
            logits,
            attn_metadata,
            spec_metadata,
            draft_model,
            resource_manager,
        )
        batch_size = attn_metadata.num_seqs
        output["accepted_draft_tokens_indices"] = self._accepted_draft_indices_tensor[:batch_size]

        return output

    @override
    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        """Override to handle dynamic tree verification."""
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        return self._sample_and_accept_dynamic_tree(
            logits, attn_metadata, spec_metadata, batch_size, num_contexts, num_gens
        )

    @override
    def prepare_1st_drafter_inputs(
        self,
        input_ids,
        position_ids,
        hidden_states,
        accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
    ):
        """Override to re-pack gen inputs from tree-topology to uniform-padded layout.

        For gen requests, the target model outputs are in tree-topology order with
        tokens_per_gen_step tokens per request. This method compresses them into
        max_draft_len + 1 tokens per request: contiguous accepted tokens + zero padding.
        This is needed because the one-model shared KV cache requires uniform padding.

        input_ids come directly from the verify kernel's accept_token output (already
        contiguous + zero-padded). position_ids are sequential from each request's base
        position. hidden_states are gathered from tree positions at the accepted path.
        """
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        batch_size = attn_metadata.num_seqs
        num_tokens = input_ids.shape[0]
        num_ctx_tokens = attn_metadata.num_ctx_tokens

        # Hidden states: FC-transform all tokens (same as base)
        hidden_size_up = spec_metadata.hidden_size * len(spec_metadata.layers_to_capture)
        hidden_states = spec_metadata.hidden_states[:num_tokens, :hidden_size_up]
        hidden_states = draft_model.apply_eagle3_fc(hidden_states)

        # Context input_ids (same as base)
        input_ids_ctx = self._prepare_context_input_ids(
            input_ids,
            num_ctx_tokens,
            spec_metadata.gather_ids,
            accepted_tokens,
            num_contexts,
        )

        if num_gens > 0:
            max_path_len = self.max_draft_len + 1
            n_acc_all = self._last_num_accepted[num_contexts:batch_size]

            # input_ids: directly from verify kernel (contiguous + zero-padded)
            input_ids_gen = self._accept_token[:num_gens].flatten().to(input_ids.dtype)

            # position_ids: sequential from base position. Padding positions get
            # base+j (sequential), same as linear tree's unaccepted draft tokens.
            gen_starts = (
                num_ctx_tokens + self._arange_max_batch[:num_gens] * self.tokens_per_gen_step
            )
            base_positions = position_ids[gen_starts]
            arange_path = self._arange_max_path[:max_path_len]
            pos_gen = (base_positions.unsqueeze(1) + arange_path.unsqueeze(0)).reshape(-1)

            hs_gen_buf = self._gather_accepted_hidden_states(
                hidden_states, gen_starts, num_gens, num_contexts, batch_size, max_path_len
            )

            # Update seq_lens for gen requests (from tokens_per_gen_step to max_path_len)
            attn_metadata._seq_lens[num_contexts:batch_size].fill_(max_path_len)
            attn_metadata._seq_lens_cuda[num_contexts:batch_size].fill_(max_path_len)
            attn_metadata.on_update()

            # Pre-compute gather_ids for Step 0
            self._step0_gather_ids = (
                num_ctx_tokens + self._arange_max_batch[:num_gens] * max_path_len + n_acc_all - 1
            ).long()

            num_gen_tokens = num_gens * max_path_len
            total_len = num_ctx_tokens + num_gen_tokens

            # Replace torch.cat with pre-allocated buffer copies
            self._step0_input_ids_buf[:num_ctx_tokens].copy_(input_ids_ctx)
            self._step0_input_ids_buf[num_ctx_tokens:total_len].copy_(input_ids_gen)
            input_ids = self._step0_input_ids_buf[:total_len]

            self._step0_position_ids_buf[:num_ctx_tokens].copy_(position_ids[:num_ctx_tokens])
            self._step0_position_ids_buf[num_ctx_tokens:total_len].copy_(pos_gen)
            position_ids = self._step0_position_ids_buf[:total_len]

            # Lazy init hidden_states buf (hidden_dim unknown at __init__)
            hidden_dim = hidden_states.shape[-1]
            if (
                self._step0_hidden_states_buf is None
                or self._step0_hidden_states_buf.shape[-1] != hidden_dim
            ):
                self._step0_hidden_states_buf = torch.zeros(
                    self._step0_input_ids_buf.shape[0],
                    hidden_dim,
                    dtype=hidden_states.dtype,
                    device="cuda",
                )
            self._step0_hidden_states_buf[:num_ctx_tokens].copy_(hidden_states[:num_ctx_tokens])
            self._step0_hidden_states_buf[num_ctx_tokens:total_len].copy_(hs_gen_buf)
            hidden_states = self._step0_hidden_states_buf[:total_len]
        else:
            # Context-only (warmup). No gen tokens.
            input_ids = input_ids_ctx
            self._step0_gather_ids = torch.empty(0, dtype=torch.long, device="cuda")

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

    def _gather_accepted_hidden_states(
        self,
        hidden_states: torch.Tensor,
        gen_starts: torch.Tensor,
        num_gens: int,
        num_contexts: int,
        batch_size: int,
        max_path_len: int,
    ) -> torch.Tensor:
        """Gather hidden states from tree-topology order into contiguous accepted-path order.

        Maps each accepted position j to its tree position via _accepted_draft_indices_tensor,
        then gathers into a pre-allocated [num_gens * max_path_len, hidden_dim] buffer.
        Padding positions (beyond n_acc) map to the root (position 0), matching the
        two-model approach. This is safe because causal mask prevents non-padding tokens
        from attending to padding tokens, and padding output is discarded by gather_ids.
        """
        hidden_dim = hidden_states.shape[-1]
        if self._step0_hs_buf is None or self._step0_hs_buf.shape[1] != hidden_dim:
            max_buf = self._arange_max_batch.shape[0] * max_path_len
            self._step0_hs_buf = torch.zeros(
                max_buf, hidden_dim, dtype=hidden_states.dtype, device="cuda"
            )
        hs_gen_buf = self._step0_hs_buf[: num_gens * max_path_len]

        # path_pos_2d[i, j]: tree position of the j-th accepted token for gen request i
        # [i, 0] = 0 (root, from buffer init), [i, j>=1] = accepted_draft_indices[i, j-1] + 1
        path_pos_2d = self._step0_path_pos_buf[:num_gens, :max_path_len]
        if max_path_len > 1:
            # accepted_draft_indices: valid positions are >= 0, padding is -1 (sentinel).
            # +1 maps: valid → tree position (1-based), padding → -1+1=0 (root).
            # So padding naturally maps to root without needing clamp.
            path_pos_2d[:, 1:] = (
                self._accepted_draft_indices_tensor[
                    num_contexts:batch_size, : max_path_len - 1
                ].long()
                + 1
            )

        src_flat = (gen_starts.unsqueeze(1) + path_pos_2d).reshape(-1)
        hs_gen_buf[:] = hidden_states[src_flat]
        return hs_gen_buf

    # ---- Dynamic tree draft loop ----

    @override
    def _forward_draft_loop(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager,
        num_contexts,
        num_gens,
        batch_size,
        num_accepted_tokens,
        original_all_rank_num_tokens,
        resource_manager,
    ):
        """Dynamic tree draft loop with growing context.

        Dynamic tree draft loop structure:
        - Step 0: initial forward, sample K tokens, update buffers
        - prepare_for_generation(0): set up KV/position/mask
        - Steps 1+: growing context forward, sample, update, prepare
        """

        self._ensure_spec_tree_manager(resource_manager)
        spec_tree_manager = self.spec_tree_manager

        assert batch_size <= self._max_batch_size, (
            f"batch_size {batch_size} exceeds pre-allocated max_batch_size {self._max_batch_size}"
        )

        # === Step 0: Initial forward ===
        # Inputs already in uniform-padded layout from prepare_1st_drafter_inputs:
        # max_draft_len + 1 tokens per gen request (contiguous accepted + zero padding).
        num_step0_tokens = self.max_draft_len + 1

        # gather_ids: use pre-allocated buffer instead of torch.cat
        step0_gather_len = self._step0_gather_ids.shape[0]
        total_gather = num_contexts + step0_gather_len
        self._gather_ids_buf[:num_contexts].copy_(spec_metadata.gather_ids[:num_contexts])
        if step0_gather_len > 0:
            self._gather_ids_buf[num_contexts:total_gather].copy_(self._step0_gather_ids)
        gather_ids = self._gather_ids_buf[:total_gather]

        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        # Spec decoding: uniform causal mask for gen requests
        # All [num_contexts:batch_size] writes are empty-slice no-ops when num_gens == 0
        # Guard: spec_decoding tensors are None during prefill-only warmup
        if attn_metadata.spec_decoding_generation_lengths is not None:
            attn_metadata.spec_decoding_generation_lengths[num_contexts:batch_size].fill_(
                num_step0_tokens
            )

            tokens_per_req = self.tokens_per_gen_step
            total_po_size = attn_metadata.spec_decoding_position_offsets.shape[0]
            max_reqs = total_po_size // tokens_per_req
            pos_2d = attn_metadata.spec_decoding_position_offsets.view(max_reqs, tokens_per_req)
            pos_2d[num_contexts:batch_size, :num_step0_tokens] = self._causal_offs[
                :num_step0_tokens
            ]

            attn_metadata.spec_decoding_packed_mask[num_contexts:batch_size].fill_(0)
            attn_metadata.spec_decoding_packed_mask[
                num_contexts:batch_size, :num_step0_tokens, 0
            ] = self._step0_causal_mask[:num_step0_tokens]

        attn_metadata.use_spec_decoding = num_gens > 0

        # Pre-step-0 KV correction: convert from target tree-width to draft
        # accepted-path width.  The target model processed tokens_per_gen_step
        # tokens per gen request, but the draft loop step 0 should only attend
        # to max_path_len (= max_draft_len + 1) accepted-path tokens.
        if num_gens > 0 and hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= self.tokens_per_gen_step - (
                self.max_draft_len + 1
            )

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            hidden_states, hidden_states_to_save = draft_model.model(**inputs)

            # Use draft model's pre-norm output as step0_hs (matches two-model
            # where Eagle3DecoderLayer writes MLP_out + residual to shared buffer).
            hs_dim = spec_metadata.hidden_size
            step0_hs = hidden_states_to_save[gather_ids]

            logits = draft_model.logits_processor(
                hidden_states[gather_ids], draft_model.lm_head, attn_metadata, True
            )

            new_draft_tokens, new_draft_scores = self.sample(
                logits, self.K, draft_model=draft_model
            )

            previous_draft_scores = self.update_draft_tokens_and_scores(
                cur_draft_idx=0,
                new_draft_tokens=new_draft_tokens,
                new_draft_scores=new_draft_scores,
                previous_draft_scores=None,
                attn_metadata=attn_metadata,
                spec_tree_manager=spec_tree_manager,
            )

            self.update_hidden_states(
                cur_draft_idx=0,
                batch_size=batch_size,
                step0_hs=step0_hs,
                hs_dim=hs_dim,
                hidden_states_to_save=None,
                selected_parents=None,
            )

            self.prepare_for_generation(
                cur_draft_idx=0,
                attn_metadata=attn_metadata,
                inputs=inputs,
                gather_ids=gather_ids,
                batch_size=batch_size,
                num_contexts=num_contexts,
                num_gens=num_gens,
                num_accepted_tokens=num_accepted_tokens,
                original_all_rank_num_tokens=original_all_rank_num_tokens,
            )

            for layer_idx in range(1, self.max_draft_len):
                num_tokens_per_req = layer_idx * self.K

                if original_all_rank_num_tokens is not None:
                    if spec_metadata.all_rank_num_seqs is not None:
                        attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_seqs

                # Growing context: process ALL accumulated tokens
                num_infer_tokens = batch_size * num_tokens_per_req

                inp_hs = self._accumulated_hs[:batch_size, :num_tokens_per_req, :].reshape(
                    num_infer_tokens, -1
                )
                inp_ids = (
                    self.draft_tokens_buffer[:batch_size, :num_tokens_per_req]
                    .reshape(-1)
                    .to(torch.int32)
                )
                inp_pos = self.position_ids_buffer[:batch_size, :num_tokens_per_req].reshape(-1)
                inputs = {
                    "input_ids": inp_ids,
                    "position_ids": inp_pos,
                    "hidden_states": inp_hs,
                    "attn_metadata": attn_metadata,
                    "spec_metadata": spec_metadata,
                }

                hidden_states, hidden_states_to_save = draft_model.model(**inputs)

                # Take last K logits per request
                hs_reshaped = hidden_states.reshape(batch_size, num_tokens_per_req, -1)
                selected_hs = hs_reshaped[:, -self.K :, :].reshape(batch_size * self.K, -1)
                logits = draft_model.logits_processor(
                    selected_hs, draft_model.lm_head, attn_metadata, True
                )

                new_draft_tokens, new_draft_scores = self.sample(
                    logits, self.K, draft_model=draft_model
                )

                # Reshape for update: [batch_size, K, K]
                new_draft_tokens = new_draft_tokens.reshape(batch_size, self.K, self.K)
                new_draft_scores = new_draft_scores.reshape(batch_size, self.K, self.K)

                previous_draft_scores = self.update_draft_tokens_and_scores(
                    cur_draft_idx=layer_idx,
                    new_draft_tokens=new_draft_tokens,
                    new_draft_scores=new_draft_scores,
                    previous_draft_scores=previous_draft_scores,
                    attn_metadata=attn_metadata,
                    spec_tree_manager=spec_tree_manager,
                )

                self.update_hidden_states(
                    cur_draft_idx=layer_idx,
                    batch_size=batch_size,
                    step0_hs=None,
                    hs_dim=hs_dim,
                    hidden_states_to_save=hidden_states_to_save,
                    selected_parents=self._last_selected_parents,
                )

                self.prepare_for_generation(
                    cur_draft_idx=layer_idx,
                    attn_metadata=attn_metadata,
                    inputs=None,
                    gather_ids=None,
                    batch_size=batch_size,
                    num_contexts=num_contexts,
                    num_gens=num_gens,
                    num_accepted_tokens=None,
                    original_all_rank_num_tokens=None,
                )

        # Resample final tokens and build tree
        real_draft_tokens, topk_score_indices = self.resampling_final_draft_tokens(batch_size)

        # Build the dynamic tree structure
        if spec_tree_manager is not None:
            self.tree_ops_converter.build_dynamic_tree(
                history_draft_tokens_parent_buffer=self.history_draft_tokens_parent_buffer[
                    :batch_size
                ],
                topk_score_indices=topk_score_indices,
                tree_mask=spec_tree_manager.spec_dec_packed_mask[:batch_size],
                positions=spec_tree_manager.spec_dec_position_offsets[:batch_size],
                retrieve_index=spec_tree_manager.retrieve_index[:batch_size],
                retrieve_next_token=spec_tree_manager.retrieve_next_token[:batch_size],
                retrieve_next_sibling=spec_tree_manager.retrieve_next_sibling[:batch_size],
                use_packed_mask=True,
            )

        return real_draft_tokens.to(torch.int32)

    # ---- Dynamic tree verification ----

    def _sample_and_accept_dynamic_tree(
        self, logits, attn_metadata, spec_metadata, batch_size, num_contexts, num_gens
    ):
        """Dynamic tree verification using CUDA kernel.

        All kernel pre/post-processing is vectorized (no Python for-loops)
        to avoid GPU-CPU sync overhead from per-request .item() calls.
        Uses pre-allocated buffers for CUDA graph compatibility.
        """
        max_total = self.max_total_draft_tokens
        N = self.tokens_per_gen_step  # includes root
        max_path_len = self.max_draft_len + 1  # max accepted path (tree depth + root)

        # Use pre-allocated return buffers
        self._accepted_tokens_buf[:batch_size].zero_()
        accepted_tokens = self._accepted_tokens_buf[:batch_size, :max_path_len]
        self._num_accepted_tokens_buf[:batch_size].fill_(1)
        num_accepted_tokens = self._num_accepted_tokens_buf[:batch_size]
        # Reset pre-allocated accepted draft indices buffer
        self._accepted_draft_indices_tensor[:batch_size].fill_(-1)

        # Sample all target tokens into pre-allocated buffer
        num_flat_tokens = logits.shape[0]
        torch.argmax(logits, dim=-1, out=self._target_tokens_buf[:num_flat_tokens])
        target_tokens = self._target_tokens_buf[:num_flat_tokens]

        # Context requests: accept sampled token
        accepted_tokens[:num_contexts, 0] = target_tokens[:num_contexts].to(torch.int32)

        # Generation requests: tree verification
        if num_gens > 0:
            spec_tree_manager = self.spec_tree_manager

            # Build target_predict into pre-allocated buffer: [num_gens, N]
            target_predict = self._target_predict_buf[:num_gens]
            target_predict[:] = target_tokens[num_contexts:].reshape(num_gens, N)

            # Build candidates into pre-allocated buffer: [num_gens, N]
            candidates = self._candidates_buf[:num_gens]
            candidates.zero_()
            candidates[:, 1:] = spec_metadata.draft_tokens.reshape(num_gens, max_total).to(
                torch.int64
            )
            candidates[:, 0] = target_predict[:, 0]

            # Call in-place verification kernel with pre-allocated output buffers
            _, accept_index, accept_token_num, accept_token = (
                self.tree_ops_converter.verify_dynamic_tree_greedy_out(
                    candidates,
                    spec_tree_manager.retrieve_index[:num_gens],
                    spec_tree_manager.retrieve_next_token[:num_gens],
                    spec_tree_manager.retrieve_next_sibling[:num_gens],
                    target_predict,
                    num_gens,
                    self.max_draft_len + 1,
                )
            )

            # Store kernel's contiguous accepted tokens for step 0 input_ids
            self._accept_token = accept_token

            # Process kernel results (vectorized, no Python for-loops)
            n_acc_draft = accept_token_num[:num_gens]  # draft-only count per request
            num_accepted_tokens[num_contexts:batch_size] = (n_acc_draft + 1).to(torch.int32)

            # accept_token/accept_index shape = [num_gens, max_path_len] (from kernel)
            # Kernel zeros buffers; only writes positions 0..n_acc, rest stays 0.

            # Accepted tokens: direct copy (buffer width = max_path_len)
            accepted_tokens[num_contexts:batch_size] = accept_token[:num_gens].to(torch.int32)

            # Accepted draft indices: 0 - 1 = -1 = sentinel, so direct copy works
            self._accepted_draft_indices_tensor[num_contexts:batch_size] = (
                accept_index[:num_gens, 1:max_path_len] - 1
            ).to(torch.int32)

        num_accepted_tokens = self._apply_force_accepted_tokens(num_accepted_tokens, num_contexts, self.max_draft_len)
        self._last_num_accepted = num_accepted_tokens

        return accepted_tokens, num_accepted_tokens

    # ---- Dynamic tree helper methods (matching two-model naming) ----

    def sample(
        self, logits: torch.Tensor, max_top_k: int, draft_model=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TopK sampling with log softmax for dynamic tree."""
        last_p = torch.log_softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(last_p, k=max_top_k, dim=-1)
        # Apply draft-to-target vocab mapping if the draft model has it
        if draft_model is not None and hasattr(draft_model.model, "d2t"):
            d2t = draft_model.model.d2t.data
            topk_indices = topk_indices + d2t[topk_indices]
        return topk_indices, topk_values

    def update_draft_tokens_and_scores(
        self,
        cur_draft_idx: int,
        new_draft_tokens: torch.Tensor,
        new_draft_scores: torch.Tensor,
        previous_draft_scores: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_tree_manager: "SpecTreeManager",
    ):
        """Update draft tokens and scores, write contiguously to buffer."""
        return_draft_scores = None
        batch_size = attn_metadata.num_seqs
        if cur_draft_idx == 0:
            new_draft_scores = new_draft_scores.reshape(batch_size, self.K)

            new_draft_tokens_2d = new_draft_tokens.reshape(batch_size, self.K)
            self.draft_tokens_buffer[:batch_size, : self.K] = new_draft_tokens_2d
            self.history_draft_tokens_buffer[:batch_size, : self.K] = new_draft_tokens_2d
            self.history_score_buffer[:batch_size, : self.K] = new_draft_scores

            # Initialize parent buffer: -1 for root, 0..K-1 for first layer
            self.history_draft_tokens_parent_buffer[:batch_size, : self.K + 1] = (
                self._parent_init_arange
            )

            self.prepare_tree_mask_and_position_offset(
                cur_draft_idx, attn_metadata, spec_tree_manager, None
            )

            return_draft_scores = new_draft_scores
        else:
            new_draft_tokens = new_draft_tokens.reshape(batch_size, self.K * self.K)

            # Accumulate scores from previous layer
            new_draft_scores = new_draft_scores + previous_draft_scores.unsqueeze(2)
            new_draft_scores = new_draft_scores.reshape(batch_size, self.K * self.K)

            # Select best K from K*K candidates
            topk_values, topk_indices = torch.topk(new_draft_scores, k=self.K, dim=-1)
            real_draft_tokens = torch.gather(new_draft_tokens, dim=1, index=topk_indices)
            num_tokens_previous_layer = cur_draft_idx * self.K
            num_tokens_current_layer = (cur_draft_idx + 1) * self.K
            self.draft_tokens_buffer[
                :batch_size, num_tokens_previous_layer:num_tokens_current_layer
            ] = real_draft_tokens

            # Save all K*K candidates to history buffers
            write_history_start_offset = self.K + (cur_draft_idx - 1) * self.K * self.K
            write_history_end_offset = write_history_start_offset + self.K * self.K
            self.history_draft_tokens_buffer[
                :batch_size, write_history_start_offset:write_history_end_offset
            ] = new_draft_tokens
            self.history_score_buffer[
                :batch_size, write_history_start_offset:write_history_end_offset
            ] = new_draft_scores

            # Determine which parents were selected
            selected_parents = topk_indices // self.K
            self._last_selected_parents = selected_parents
            self.prepare_tree_mask_and_position_offset(
                cur_draft_idx, attn_metadata, spec_tree_manager, selected_parents
            )

            # Update parent buffer for next layer
            if cur_draft_idx < self.max_draft_len - 1:
                next_layer_start = cur_draft_idx * self.K + 1
                next_layer_end = next_layer_start + self.K
                parents_relative_indices = topk_indices + self.K**2 * (cur_draft_idx - 1) + self.K
                self.history_draft_tokens_parent_buffer[
                    :batch_size, next_layer_start:next_layer_end
                ] = parents_relative_indices

            return_draft_scores = topk_values
        return return_draft_scores

    def resampling_final_draft_tokens(self, batch_size: int):
        """Reconstruct the tree based on history buffers."""
        topk_score_indices = torch.topk(
            self.history_score_buffer[:batch_size, :], k=self.max_total_draft_tokens, dim=-1
        ).indices
        topk_score_indices = torch.sort(topk_score_indices).values

        real_draft_tokens = torch.gather(
            self.history_draft_tokens_buffer[:batch_size, :], dim=1, index=topk_score_indices
        )

        return real_draft_tokens, topk_score_indices

    def prepare_tree_mask_and_position_offset(
        self,
        cur_draft_idx: int,
        attn_metadata: AttentionMetadata,
        spec_tree_manager: SpecTreeManager,
        selected_parents: torch.Tensor = None,
    ):
        """Prepare the mask and position offsets for the next layer."""
        if attn_metadata.spec_decoding_packed_mask is None:
            return
        batch_size = attn_metadata.num_seqs
        num_tokens_current_layer = self.K * (cur_draft_idx + 1)
        num_tokens_previous_layer = self.K * cur_draft_idx
        if cur_draft_idx == 0:
            attn_metadata.spec_decoding_packed_mask.fill_(0)
            spec_tree_manager.compute_spec_dec_packed_mask(
                self.tree_mask_init_buffer[:batch_size],
                attn_metadata.spec_decoding_packed_mask[:batch_size],
            )
            self.tree_mask_buffer[
                : batch_size * num_tokens_current_layer * num_tokens_current_layer
            ].copy_(self.tree_mask_init_buffer[:batch_size].view(-1))
            attn_metadata.spec_decoding_position_offsets.fill_(0)
            attn_metadata.spec_decoding_generation_lengths[:batch_size] = num_tokens_current_layer
        else:
            num_parent_mask = batch_size * cur_draft_idx * self.K * cur_draft_idx * self.K
            parent_mask = self.tree_mask_buffer[:num_parent_mask].reshape(
                batch_size, cur_draft_idx * self.K, cur_draft_idx * self.K
            )

            selected_parents_expanded = selected_parents.unsqueeze(-1).expand(
                batch_size, self.K, parent_mask.size(-1)
            )
            parent_mask_selected = torch.gather(
                parent_mask[:, -self.K :, :], dim=1, index=selected_parents_expanded
            )

            # Replace torch.cat with pre-allocated buffer copies for current_mask
            # current_mask = cat([parent_mask_selected, tree_mask_init], dim=2) then
            # current_mask = cat([mask_padding, current_mask], dim=1)
            current_mask = self._current_mask_buf[
                :batch_size, :num_tokens_current_layer, :num_tokens_current_layer
            ]
            # Top rows: padding zeros
            current_mask[:, :num_tokens_previous_layer, :].zero_()
            # Bottom rows: [parent_mask_selected | tree_mask_init]
            prev_cols = parent_mask.size(-1)  # = num_tokens_previous_layer
            current_mask[:, num_tokens_previous_layer:, :prev_cols].copy_(parent_mask_selected)
            current_mask[:, num_tokens_previous_layer:, prev_cols:num_tokens_current_layer].copy_(
                self.tree_mask_init_buffer[:batch_size]
            )

            spec_tree_manager.compute_spec_dec_packed_mask(
                current_mask, attn_metadata.spec_decoding_packed_mask[:batch_size]
            )
            self.tree_mask_buffer[
                : batch_size * num_tokens_current_layer * num_tokens_current_layer
            ].copy_(current_mask.reshape(-1))

            attn_metadata.spec_decoding_generation_lengths[:batch_size] = num_tokens_current_layer

            # Replace torch.cat for position offsets with pre-allocated buffer
            previous_position_offsets = attn_metadata.spec_decoding_position_offsets[
                : batch_size * num_tokens_previous_layer
            ].view(batch_size, num_tokens_previous_layer)

            new_pos = self._new_pos_offset_buf[:batch_size, :num_tokens_current_layer]
            new_pos[:, :num_tokens_previous_layer].copy_(previous_position_offsets)
            new_pos[:, num_tokens_previous_layer:num_tokens_current_layer].copy_(
                previous_position_offsets[:, -self.K :] + 1
            )
            attn_metadata.spec_decoding_position_offsets[
                : batch_size * num_tokens_current_layer
            ] = new_pos.reshape(-1)

    def prepare_for_generation(
        self,
        cur_draft_idx: int,
        attn_metadata: AttentionMetadata,
        inputs,
        gather_ids,
        batch_size: int,
        num_contexts: int,
        num_gens: int,
        num_accepted_tokens,
        original_all_rank_num_tokens,
    ):
        """Set up attn_metadata for the subsequent drafter layer.

        Matches two-model prepare_for_generation() structure:
        - Step 0: position IDs, seq_lens, KV rewind, host_request_types
        - Steps 1+: extend position IDs, update seq_lens, increment kv_lens
        """
        num_tokens_current_layer = self.K * (cur_draft_idx + 1)

        if cur_draft_idx == 0:
            # Position IDs: base_pos + 1, replicated K times per batch
            base_pos = inputs["position_ids"][gather_ids] + 1
            self.position_ids_buffer[:batch_size, : self.K] = base_pos.unsqueeze(1).expand(
                -1, self.K
            )

            # KV cache: rewind to stable state then pre-add K
            attn_metadata._seq_lens[:batch_size].fill_(self.K)
            attn_metadata._seq_lens_cuda[:batch_size].fill_(self.K)
            attn_metadata.on_update()

            if inputs["attn_metadata"].kv_cache_manager is not None:
                attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
                attn_metadata.num_contexts = 0

            if hasattr(attn_metadata, "kv_lens_cuda"):
                # KV rewind: remove unaccepted draft-path tokens for gen
                # requests.  At this point kv_lens already reflects the
                # accepted-path width (max_draft_len + 1) thanks to the
                # pre-step-0 correction in _forward_draft_loop.
                # This mirrors the linear tree pattern (eagle3.py:594-597).
                if num_gens > 0:
                    attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                        self.max_draft_len + 1
                    ) - num_accepted_tokens[num_contexts:batch_size]
                attn_metadata.kv_lens_cuda[:batch_size] += self.K

            attn_metadata.use_spec_decoding = True

        else:
            # Position IDs: append prev[-K:] + 1
            num_tokens_previous_layer = cur_draft_idx * self.K
            prev_pos = self.position_ids_buffer[:batch_size, :num_tokens_previous_layer]
            self.position_ids_buffer[
                :batch_size, num_tokens_previous_layer:num_tokens_current_layer
            ] = prev_pos[:, -self.K :] + 1

            # Growing seq_lens and pre-increment kv_lens
            attn_metadata._seq_lens[:batch_size].fill_(num_tokens_current_layer)
            attn_metadata._seq_lens_cuda[:batch_size].fill_(num_tokens_current_layer)
            attn_metadata.on_update()
            attn_metadata.kv_lens_cuda[:batch_size] += self.K

    def update_hidden_states(
        self,
        cur_draft_idx: int,
        batch_size: int,
        step0_hs,
        hs_dim: int,
        hidden_states_to_save=None,
        selected_parents=None,
    ):
        """Manage hidden states for the growing context pattern.

        One-model uses accumulated_hs, hs_write_buffer, hs_read_map to replicate
        the two-model's resource-manager-based hidden state tracking:
        - Step 0: initialize buffers from step0_hs (draft model pre-norm at last accepted token)
        - Steps 1+: write prenorm to buffer, set read_map via selected_parents, reconstruct
        """
        if cur_draft_idx == 0:
            self._hs_dim = hs_dim

            # Lazy init _hs_write_buffer and _accumulated_hs (hs_dim unknown at __init__)
            if self._hs_write_buffer is None or self._hs_write_buffer.shape[2] != hs_dim:
                self._hs_write_buffer = torch.zeros(
                    self._arange_max_batch.shape[0],
                    self.max_draft_len * self.K,
                    hs_dim,
                    device=step0_hs.device,
                    dtype=step0_hs.dtype,
                )
            if self._accumulated_hs is None or self._accumulated_hs.shape[2] != hs_dim:
                self._accumulated_hs = torch.zeros(
                    self._arange_max_batch.shape[0],
                    self.max_draft_len * self.K,
                    hs_dim,
                    device=step0_hs.device,
                    dtype=step0_hs.dtype,
                )

            # All K depth-0 tokens share the draft model pre-norm at last
            # accepted token (matches two-model where start_idx reads from
            # resource manager)
            self._accumulated_hs[:batch_size, : self.K] = step0_hs.unsqueeze(1).expand(
                -1, self.K, -1
            )
            self._step0_hs = step0_hs

        else:
            num_tokens_per_req = cur_draft_idx * self.K

            hs_to_save_reshaped = hidden_states_to_save.reshape(batch_size, num_tokens_per_req, -1)

            # 1) Write current forward's prenorm to write_buffer
            self._hs_write_buffer[:batch_size, :num_tokens_per_req] = hs_to_save_reshaped

            # 2) Set read_map for new K tokens (depth cur_draft_idx)
            parent_offset = (cur_draft_idx - 1) * self.K
            self._hs_read_map[
                :batch_size, cur_draft_idx * self.K : (cur_draft_idx + 1) * self.K
            ] = parent_offset + selected_parents

            # 3) Reconstruct accumulated_hs in pre-allocated buffer:
            #    first K = step0_hs, rest gathered from write_buffer
            num_tokens_next = (cur_draft_idx + 1) * self.K
            self._accumulated_hs[:batch_size, : self.K] = self._step0_hs.unsqueeze(1).expand(
                -1, self.K, -1
            )
            if num_tokens_next > self.K:
                read_idx = self._hs_read_map[:batch_size, self.K : num_tokens_next]
                gathered = torch.gather(
                    self._hs_write_buffer[:batch_size],
                    1,
                    read_idx.unsqueeze(-1).expand(-1, -1, self._hs_dim),
                )
                self._accumulated_hs[:batch_size, self.K : num_tokens_next] = gathered
