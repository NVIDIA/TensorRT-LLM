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
"""Eagle3 one-model dynamic tree speculative decoding."""

from typing import TYPE_CHECKING, override

import torch

from ..attention_backend import AttentionMetadata
from .eagle3 import Eagle3OneModelWorker
from .spec_tree_manager import SpecTreeManager

if TYPE_CHECKING:
    from ...llmapi.llm_args import EagleDecodingConfig


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

        self.K = spec_config.dynamic_tree_max_topK
        self.max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
        self.tokens_per_gen_step = self.max_total_draft_tokens + 1
        self._max_batch_size = spec_config.runtime_max_batch_size or 256

        K = self.K
        max_draft_len = spec_config.max_draft_len
        max_total_draft_tokens = self.max_total_draft_tokens
        max_batch_size = self._max_batch_size

        # Pre-allocated 2D buffers
        self.draft_tokens_buffer = torch.zeros(
            max_batch_size, self.max_total_draft_tokens, dtype=torch.int64, device="cuda"
        )
        self.position_ids_buffer = torch.zeros(
            max_batch_size, self.tokens_per_gen_step, dtype=torch.int64, device="cuda"
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
            (max_batch_size * self.tokens_per_gen_step * self.tokens_per_gen_step),
            dtype=torch.int32,
            device="cuda",
        )
        self.tree_mask_init_buffer = (
            torch.eye(K, dtype=torch.int32, device="cuda").unsqueeze(0).repeat(max_batch_size, 1, 1)
        )
        self.tree_ops_converter = DynamicTreeOpsConverter(
            dynamic_tree_max_topK=K,
            max_draft_len=max_draft_len,
            max_total_draft_tokens=self.max_total_draft_tokens,
            max_batch_size=max_batch_size,
            device=torch.device("cuda"),
        )

        self._accepted_draft_indices_tensor = torch.full(
            (max_batch_size, max_draft_len), -1, dtype=torch.int32, device="cuda"
        )

        # Initialized by _sample_and_accept_dynamic_tree; None during warmup.
        self._accept_token = None
        self._last_num_accepted = None
        self._last_selected_parents = None
        self._kv_byte_size = None  # Lazily set on first _relocate_kv_eagerly call

        # Pre-allocated buffers for _relocate_kv_eagerly (avoid per-call allocations)
        self._reloc_n_acc_draft = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")
        self._reloc_offsets = torch.zeros(max_batch_size + 1, dtype=torch.int32, device="cuda")
        self._reloc_col_idx = torch.arange(max_draft_len, device="cuda")
        self._reloc_valid_mask = torch.zeros(
            max_batch_size, max_draft_len, dtype=torch.bool, device="cuda"
        )
        self._reloc_rewind_adj = torch.zeros(max_batch_size, dtype=torch.int32, device="cuda")

        # Hidden state management buffers (lazily initialized in update_hidden_states)
        self._hs_write_buffer = None
        self._hs_read_map = torch.zeros(
            max_batch_size, max_draft_len * K, dtype=torch.long, device="cuda"
        )
        self._accumulated_hs = None
        self._step0_hs = None
        self._hs_dim = None

        # Step 0 buffers
        max_step0_tokens = max_draft_len + 1
        self._step0_hs_buf = None
        self._step0_causal_mask = torch.tensor(
            [(1 << (t + 1)) - 1 for t in range(max_step0_tokens)],
            dtype=torch.int64,
            device="cuda",
        )
        self._arange_max_batch = torch.arange(max_batch_size, device="cuda")
        self._arange_max_path = torch.arange(max_step0_tokens, device="cuda")
        self._causal_offs = torch.arange(max_step0_tokens, device="cuda", dtype=torch.int32)
        self._step0_path_pos_buf = torch.zeros(
            max_batch_size, max_step0_tokens, dtype=torch.long, device="cuda"
        )
        self._step0_gather_ids = torch.empty(0, dtype=torch.long, device="cuda")
        self._parent_init_arange = torch.arange(-1, K, device="cuda", dtype=torch.int32)

        # Verification buffers
        max_path_len = max_draft_len + 1
        tokens_per_gen_step = self.tokens_per_gen_step
        self._accepted_tokens_buf = torch.zeros(
            max_batch_size, max_path_len, dtype=torch.int32, device="cuda"
        )
        self._num_accepted_tokens_buf = torch.ones(max_batch_size, dtype=torch.int32, device="cuda")
        self._target_tokens_buf = torch.zeros(
            max_batch_size * tokens_per_gen_step, dtype=torch.int64, device="cuda"
        )
        self._candidates_buf = torch.zeros(
            max_batch_size, tokens_per_gen_step, dtype=torch.int64, device="cuda"
        )
        self._target_predict_buf = torch.zeros(
            max_batch_size, tokens_per_gen_step, dtype=torch.int64, device="cuda"
        )

        # Step 0 input buffers
        max_total_tokens = max_batch_size * tokens_per_gen_step
        self._step0_input_ids_buf = torch.zeros(max_total_tokens, dtype=torch.int64, device="cuda")
        self._step0_position_ids_buf = torch.zeros(
            max_total_tokens, dtype=torch.int64, device="cuda"
        )
        self._step0_hidden_states_buf = None
        self._gather_ids_buf = torch.zeros(max_total_tokens, dtype=torch.long, device="cuda")
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

    def _relocate_kv_eagerly(self, attn_metadata, batch_size):
        """Move accepted draft tokens' KV from tree to linear positions."""
        cache_mgr = getattr(attn_metadata, "kv_cache_manager", None)
        if self._last_num_accepted is None or cache_mgr is None:
            return

        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        if num_gens <= 0:
            return

        # Accepted DRAFT tokens per request (exclude root), using pre-allocated buffer
        n_acc_draft = self._reloc_n_acc_draft[:batch_size]
        n_acc_draft.copy_(self._last_num_accepted[:batch_size])
        n_acc_draft.sub_(1).clamp_(min=0)
        n_acc_draft[:num_contexts].zero_()

        # Build packed offsets using pre-allocated buffer
        offsets = self._reloc_offsets[: batch_size + 1]
        offsets[0] = 0
        torch.cumsum(n_acc_draft, dim=0, out=offsets[1:])

        indices_2d = self._accepted_draft_indices_tensor[:batch_size]
        max_draft = indices_2d.shape[1]
        col_idx = self._reloc_col_idx[:max_draft]
        valid_mask = self._reloc_valid_mask[:batch_size, :max_draft]
        torch.less(col_idx.unsqueeze(0), n_acc_draft.unsqueeze(1), out=valid_mask)
        packed = indices_2d[valid_mask]

        if packed.numel() == 0:
            return

        past_kv_lens = attn_metadata.kv_lens_cuda[:batch_size]

        if self._kv_byte_size is None:
            from tensorrt_llm.bindings import DataType

            dtype = cache_mgr.dtype
            if dtype in (DataType.HALF, DataType.BF16):
                self._kv_byte_size = 2.0
            elif dtype == DataType.FLOAT:
                self._kv_byte_size = 4.0
            elif dtype in (DataType.FP8, DataType.INT8):
                self._kv_byte_size = 1.0
            else:
                self._kv_byte_size = 0.5  # INT4

        rewind_adj = self._reloc_rewind_adj[:batch_size]
        rewind_adj.zero_()

        torch.ops.tensorrt_llm.update_kv_cache_draft_token_location(
            offsets,
            packed,
            past_kv_lens,
            True,  # use_paged_kv_cache
            cache_mgr.num_layers,
            cache_mgr.num_kv_heads,
            int(cache_mgr.head_dim * self._kv_byte_size),
            cache_mgr.max_total_draft_tokens,
            cache_mgr.max_attention_window_vec[0],
            rewind_adj,
            None,
            cache_mgr.kv_cache_pool_pointers,
            attn_metadata.kv_cache_block_offsets,
            cache_mgr.max_blocks_per_seq,
            cache_mgr.tokens_per_block,
            None,
        )

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
        """Re-pack gen inputs from tree-topology to uniform-padded layout
        (max_draft_len + 1 tokens per gen request) for the shared KV cache.
        """
        num_contexts = attn_metadata.num_contexts
        num_gens = attn_metadata.num_seqs - num_contexts
        batch_size = attn_metadata.num_seqs
        num_tokens = input_ids.shape[0]
        num_ctx_tokens = attn_metadata.num_ctx_tokens

        hidden_size_up = spec_metadata.hidden_size * len(spec_metadata.layers_to_capture)
        hidden_states = spec_metadata.hidden_states[:num_tokens, :hidden_size_up]
        hidden_states = draft_model.apply_eagle3_fc(hidden_states)

        input_ids_ctx = self._prepare_context_input_ids(
            input_ids,
            num_ctx_tokens,
            spec_metadata.gather_ids,
            accepted_tokens,
            num_contexts,
        )

        if num_gens > 0:
            max_path_len = self.max_draft_len + 1

            # CUDA graph warmup: verification hasn't run yet
            if self._last_num_accepted is None:
                self._last_num_accepted = torch.ones(
                    self._max_batch_size, dtype=torch.int32, device="cuda"
                )
            if self._accept_token is None:
                self._accept_token = torch.zeros(
                    self._max_batch_size, max_path_len, dtype=torch.int64, device="cuda"
                )

            n_acc_all = self._last_num_accepted[num_contexts:batch_size]

            input_ids_gen = self._accept_token[:num_gens].flatten().to(input_ids.dtype)
            gen_starts = (
                num_ctx_tokens + self._arange_max_batch[:num_gens] * self.tokens_per_gen_step
            )
            base_positions = position_ids[gen_starts]
            arange_path = self._arange_max_path[:max_path_len]
            pos_gen = (base_positions.unsqueeze(1) + arange_path.unsqueeze(0)).reshape(-1)

            hs_gen_buf = self._gather_accepted_hidden_states(
                hidden_states, gen_starts, num_gens, num_contexts, batch_size, max_path_len
            )

            attn_metadata._seq_lens[num_contexts:batch_size].fill_(max_path_len)
            attn_metadata._seq_lens_cuda[num_contexts:batch_size].fill_(max_path_len)
            attn_metadata.on_update()

            self._step0_gather_ids = (
                num_ctx_tokens + self._arange_max_batch[:num_gens] * max_path_len + n_acc_all - 1
            ).long()

            num_gen_tokens = num_gens * max_path_len
            total_len = num_ctx_tokens + num_gen_tokens

            self._step0_input_ids_buf[:num_ctx_tokens].copy_(input_ids_ctx)
            self._step0_input_ids_buf[num_ctx_tokens:total_len].copy_(input_ids_gen)
            input_ids = self._step0_input_ids_buf[:total_len]

            self._step0_position_ids_buf[:num_ctx_tokens].copy_(position_ids[:num_ctx_tokens])
            self._step0_position_ids_buf[num_ctx_tokens:total_len].copy_(pos_gen)
            position_ids = self._step0_position_ids_buf[:total_len]

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
            self._step0_gather_ids = self._gather_ids_buf[:0]

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
        """Gather hidden states from tree-topology to contiguous accepted-path order."""
        hidden_dim = hidden_states.shape[-1]
        if self._step0_hs_buf is None or self._step0_hs_buf.shape[1] != hidden_dim:
            max_buf = self._arange_max_batch.shape[0] * max_path_len
            self._step0_hs_buf = torch.zeros(
                max_buf, hidden_dim, dtype=hidden_states.dtype, device="cuda"
            )
        hs_gen_buf = self._step0_hs_buf[: num_gens * max_path_len]

        # path_pos_2d[i, j]: tree position of j-th accepted token for gen request i
        path_pos_2d = self._step0_path_pos_buf[:num_gens, :max_path_len]
        path_pos_2d[:, 0] = 0  # root position is always 0
        if max_path_len > 1:
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
        """Dynamic tree draft loop with growing context."""

        self._ensure_spec_tree_manager(resource_manager)
        spec_tree_manager = self.spec_tree_manager

        assert batch_size <= self._max_batch_size, (
            f"batch_size {batch_size} exceeds pre-allocated max_batch_size {self._max_batch_size}"
        )

        # === Step 0: Initial forward ===
        num_step0_tokens = self.max_draft_len + 1

        step0_gather_len = self._step0_gather_ids.shape[0]
        total_gather = num_contexts + step0_gather_len
        self._gather_ids_buf[:num_contexts].copy_(spec_metadata.gather_ids[:num_contexts])
        if step0_gather_len > 0:
            self._gather_ids_buf[num_contexts:total_gather].copy_(self._step0_gather_ids)
        gather_ids = self._gather_ids_buf[:total_gather]

        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        # Set up uniform causal mask for step 0 (None during prefill-only warmup)
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

        # KV correction: target processed tokens_per_gen_step per request,
        # but step 0 should only attend to max_draft_len + 1 accepted-path tokens.
        if num_gens > 0 and hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= self.tokens_per_gen_step - (
                self.max_draft_len + 1
            )

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            hidden_states, hidden_states_to_save = draft_model.model(**inputs)

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
        """Dynamic tree verification using CUDA kernel."""
        N = self.tokens_per_gen_step
        max_path_len = self.max_draft_len + 1

        # Reset output buffers
        self._accepted_tokens_buf[:batch_size].zero_()
        accepted_tokens = self._accepted_tokens_buf[:batch_size, :max_path_len]
        self._num_accepted_tokens_buf[:batch_size].fill_(1)
        num_accepted_tokens = self._num_accepted_tokens_buf[:batch_size]
        self._accepted_draft_indices_tensor[:batch_size].fill_(-1)

        num_flat_tokens = logits.shape[0]
        torch.argmax(logits, dim=-1, out=self._target_tokens_buf[:num_flat_tokens])
        target_tokens = self._target_tokens_buf[:num_flat_tokens]

        # Context requests: accept sampled token
        accepted_tokens[:num_contexts, 0] = target_tokens[:num_contexts].to(torch.int32)

        # Generation requests: tree verification
        if num_gens > 0:
            spec_tree_manager = self.spec_tree_manager

            target_predict = self._target_predict_buf[:num_gens]
            target_predict[:] = target_tokens[num_contexts:].reshape(num_gens, N)

            candidates = self._candidates_buf[:num_gens]
            candidates[:, 1:] = spec_metadata.draft_tokens.reshape(num_gens, N - 1).to(torch.int64)
            candidates[:, 0] = target_predict[:, 0]

            if spec_tree_manager is None:
                # CUDA graph warmup: accept only the first token per request
                num_accepted_tokens[num_contexts:batch_size] = 1
                accepted_tokens[num_contexts:batch_size, 0] = target_predict[:, 0].to(torch.int32)
                self._accepted_draft_indices_tensor[num_contexts:batch_size] = -1
                return accepted_tokens, num_accepted_tokens

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

            self._accept_token = accept_token
            n_acc_draft = accept_token_num[:num_gens]
            num_accepted_tokens[num_contexts:batch_size] = (n_acc_draft + 1).to(torch.int32)
            accepted_tokens[num_contexts:batch_size] = accept_token[:num_gens].to(torch.int32)
            # accept_index is 0-based from kernel; -1 converts padding (0) to sentinel (-1)
            self._accepted_draft_indices_tensor[num_contexts:batch_size] = (
                accept_index[:num_gens, 1:max_path_len] - 1
            ).to(torch.int32)

        num_accepted_tokens = self._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, self.max_draft_len
        )
        self._last_num_accepted = num_accepted_tokens

        return accepted_tokens, num_accepted_tokens

    def sample(
        self, logits: torch.Tensor, max_top_k: int, draft_model=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TopK sampling with softmax for dynamic tree."""
        last_p = torch.softmax(logits, dim=-1)
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
        batch_size = attn_metadata.num_seqs
        if cur_draft_idx == 0:
            new_draft_scores = new_draft_scores.reshape(batch_size, self.K)

            new_draft_tokens_2d = new_draft_tokens.reshape(batch_size, self.K)
            self.draft_tokens_buffer[:batch_size, : self.K] = new_draft_tokens_2d
            self.history_draft_tokens_buffer[:batch_size, : self.K] = new_draft_tokens_2d
            self.history_score_buffer[:batch_size, : self.K] = new_draft_scores

            # Parent buffer: -1 for root, 0..K-1 for first layer
            self.history_draft_tokens_parent_buffer[:batch_size, : self.K + 1] = (
                self._parent_init_arange
            )

            self.prepare_tree_mask_and_position_offset(
                cur_draft_idx, attn_metadata, spec_tree_manager, None
            )

            return new_draft_scores
        else:
            new_draft_tokens = new_draft_tokens.reshape(batch_size, self.K * self.K)

            # Accumulate scores from previous layer (probability space)
            new_draft_scores = new_draft_scores * previous_draft_scores.unsqueeze(2)
            new_draft_scores = new_draft_scores.reshape(batch_size, self.K * self.K)

            # Select best K from K*K candidates
            topk_values, topk_indices = torch.topk(new_draft_scores, k=self.K, dim=-1)
            real_draft_tokens = torch.gather(new_draft_tokens, dim=1, index=topk_indices)
            num_tokens_previous_layer = cur_draft_idx * self.K
            num_tokens_current_layer = (cur_draft_idx + 1) * self.K
            self.draft_tokens_buffer[
                :batch_size, num_tokens_previous_layer:num_tokens_current_layer
            ] = real_draft_tokens

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

            return topk_values

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
        """Set up attn_metadata for the subsequent drafter layer."""
        num_tokens_current_layer = self.K * (cur_draft_idx + 1)

        if cur_draft_idx == 0:
            base_pos = inputs["position_ids"][gather_ids] + 1
            self.position_ids_buffer[:batch_size, : self.K] = base_pos.unsqueeze(1).expand(
                -1, self.K
            )

            attn_metadata._seq_lens[:batch_size].fill_(self.K)
            attn_metadata._seq_lens_cuda[:batch_size].fill_(self.K)
            attn_metadata.on_update()

            if inputs["attn_metadata"].kv_cache_manager is not None:
                attn_metadata.host_request_types[: attn_metadata.num_contexts].fill_(1)
                attn_metadata.num_contexts = 0

            if hasattr(attn_metadata, "kv_lens_cuda"):
                # KV rewind: remove unaccepted draft-path tokens
                if num_gens > 0:
                    attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                        self.max_draft_len + 1
                    ) - num_accepted_tokens[num_contexts:batch_size]
                attn_metadata.kv_lens_cuda[:batch_size] += self.K

            attn_metadata.use_spec_decoding = True

        else:
            num_tokens_previous_layer = cur_draft_idx * self.K
            prev_pos = self.position_ids_buffer[:batch_size, :num_tokens_previous_layer]
            self.position_ids_buffer[
                :batch_size, num_tokens_previous_layer:num_tokens_current_layer
            ] = prev_pos[:, -self.K :] + 1

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
        """Manage hidden states for the growing context pattern."""
        if cur_draft_idx == 0:
            self._hs_dim = hs_dim

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

            # All K depth-0 tokens share step0_hs
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
