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
"""MTP-Eagle one-model dynamic tree speculative decoding (greedy only)."""

import math
from typing import TYPE_CHECKING, List, Optional

import torch
import triton

from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._utils import get_sm_version, nvtx_range

from ..distributed.ops import allgather
from ..model_config import ModelConfig
from ..pyexecutor.llm_request import LlmRequest
from ..pyexecutor.resource_manager import BaseResourceManager
from ..pyexecutor.scheduler import ScheduledRequests
from .eagle3 import MTPEagleWorker

# Reuse drafter-agnostic dynamic-tree helpers.
from .eagle3_dynamic_tree import (
    _build_mask_and_position,
    _gather_repack_step0_kernel,
    _resample_final_tokens,
    _select_topk_draft_tokens,
)
from .mtp import MTPHiddenStatesManager

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig


class MTPEagleDynamicTreeWorker(MTPEagleWorker):
    """MTP-Eagle worker with dynamic-tree draft and greedy verify."""

    def __init__(
        self,
        spec_config: "MTPDecodingConfig",
        model_config: Optional[ModelConfig] = None,
        use_separate_draft_kv_cache: bool = False,
    ):
        super().__init__(spec_config, model_config, use_separate_draft_kv_cache)
        assert getattr(spec_config, "use_dynamic_tree", False), (
            "MTPEagleDynamicTreeWorker requires use_dynamic_tree=True"
        )

        from .dynamic_tree_ops import DynamicTreeOpsConverter

        self.K = spec_config.dynamic_tree_max_topK
        self.max_total_draft_tokens = spec_config.tokens_per_gen_step - 1
        self.tokens_per_gen_step = spec_config.tokens_per_gen_step
        # Set by py_executor_creator from the global max_batch_size.
        assert spec_config._max_batch_size is not None, (
            "MTPDecodingConfig._max_batch_size was not populated; "
            "py_executor_creator should have set it from the global max_batch_size."
        )
        self._max_batch_size = spec_config._max_batch_size

        K = self.K
        max_draft_len = spec_config.max_draft_len
        max_batch_size = self._max_batch_size
        loop_max_tokens = K * max_draft_len  # draft loop working size

        # spec_tree_manager is lazily bound from the resource manager.
        self.spec_tree_manager = None
        self._d2t = None

        # Pre-allocated draft-loop buffers (CUDA-graph safe).
        self.draft_tokens_buffer = torch.zeros(
            max_batch_size, loop_max_tokens, dtype=torch.int32, device="cuda"
        )
        self.position_ids_buffer = torch.zeros(
            max_batch_size, loop_max_tokens, dtype=torch.int32, device="cuda"
        )
        self.history_draft_tokens_buffer = torch.zeros(
            (max_batch_size, (K + K * K * (max_draft_len - 1))), dtype=torch.int32, device="cuda"
        )
        self.history_score_buffer = torch.zeros(
            (max_batch_size, K + K * K * (max_draft_len - 1)), dtype=torch.float32, device="cuda"
        )
        self.history_draft_tokens_parent_buffer = torch.zeros(
            (max_batch_size, max(K * (max_draft_len - 1) + 1, K + 1)),
            dtype=torch.int64,
            device="cuda",
        )
        self.tree_mask_buffer = torch.zeros(
            (max_batch_size * loop_max_tokens * loop_max_tokens), dtype=torch.int32, device="cuda"
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

        self._max_path_len = max_draft_len + 1
        # Step-0 draft resets verify-time tree metadata to accepted-path width.
        self._kv_correction = self.tokens_per_gen_step - self._max_path_len
        self._step0_causal_mask = torch.tensor(
            [(1 << (t + 1)) - 1 for t in range(self._max_path_len)],
            dtype=torch.int32,
            device="cuda",
        )
        self._causal_offs = torch.arange(self._max_path_len, device="cuda", dtype=torch.int32)
        self._last_selected_parents = None
        self._parent_init_arange = torch.arange(-1, K, device="cuda", dtype=torch.int32)

        # Accepted-path bookkeeping for KV relocation and output.
        self._accepted_draft_indices_tensor = torch.full(
            (max_batch_size, max_draft_len), -1, dtype=torch.int32, device="cuda"
        )
        self._kv_head_dim_bytes = None

        # === Verification buffers (greedy only) ===
        N = self.tokens_per_gen_step
        self._accepted_tokens_buf = torch.zeros(
            max_batch_size, self._max_path_len, dtype=torch.int32, device="cuda"
        )
        self._num_accepted_tokens_buf = torch.ones(max_batch_size, dtype=torch.int32, device="cuda")
        self._target_tokens_buf = torch.zeros(max_batch_size * N, dtype=torch.int64, device="cuda")
        self._candidates_buf = torch.zeros(max_batch_size, N, dtype=torch.int32, device="cuda")
        self._target_predict_buf = torch.zeros(max_batch_size, N, dtype=torch.int32, device="cuda")

        # Hidden states for the growing-context draft loop.
        self._hs_write_buffer = None
        self._accumulated_hs = None
        self._hs_read_map = torch.zeros(
            max_batch_size, loop_max_tokens, dtype=torch.long, device="cuda"
        )
        self._step0_hs = None
        self._hs_dim = None

        # Step-0 repack scratch for accepted-path inputs.
        max_total_tokens = max_batch_size * self.tokens_per_gen_step
        self._step0_input_ids_buf = torch.zeros(max_total_tokens, dtype=torch.int32, device="cuda")
        self._step0_position_ids_buf = torch.zeros(
            max_total_tokens, dtype=torch.int32, device="cuda"
        )
        self._step0_hidden_states_buf = None
        self._gather_ids_buf = torch.zeros(max_total_tokens, dtype=torch.long, device="cuda")

        # Mask repack scratch (graph-safe; avoids .contiguous() in the loop).
        buf_dim = max(self.max_total_draft_tokens + 1, loop_max_tokens)
        mask_width = (buf_dim + 31) // 32
        self._mask_repack_buf = torch.zeros(
            max_batch_size * buf_dim * mask_width, dtype=torch.int32, device="cuda"
        )
        # sm>=100 (except 120/121): prepareCustomMask keeps padded 3D; no repack.
        sm = get_sm_version()
        self._needs_mask_repack = sm < 100 or sm in (120, 121)

    def _prepare_attn_metadata_for_spec_dec(self, attn_metadata):
        super()._prepare_attn_metadata_for_spec_dec(attn_metadata)

        batch_size = attn_metadata.num_seqs
        if hasattr(attn_metadata, "kv_lens_cuda"):
            # Keep kv_lens_cuda itself alive because TRTLLM attention holds a
            # runtime view into it.
            self._saved_kv_lens_cuda = attn_metadata.kv_lens_cuda[:batch_size].clone()
        else:
            self._saved_kv_lens_cuda = None

        # Restore verify metadata after the draft loop mutates it.
        if attn_metadata.spec_decoding_packed_mask is not None:
            self._saved_packed_mask = attn_metadata.spec_decoding_packed_mask[:batch_size].clone()
        else:
            self._saved_packed_mask = None
        if attn_metadata.spec_decoding_position_offsets is not None:
            self._saved_position_offsets = attn_metadata.spec_decoding_position_offsets.clone()
            self._saved_position_offsets_cpp = attn_metadata.spec_decoding_position_offsets_cpp
        else:
            self._saved_position_offsets = None
            self._saved_position_offsets_cpp = None
        if attn_metadata.spec_decoding_generation_lengths is not None:
            self._saved_generation_lengths = attn_metadata.spec_decoding_generation_lengths[
                :batch_size
            ].clone()
        else:
            self._saved_generation_lengths = None

    def prepare_position_ids_and_last_tokens(self, position_ids, seq_lens_cuda):
        position_ids = position_ids.squeeze(0)
        last_tokens_idx = torch.cumsum(seq_lens_cuda, dim=0, dtype=torch.long) - 1
        return position_ids, last_tokens_idx

    def _restore_attn_metadata_from_spec_dec(self, attn_metadata):
        super()._restore_attn_metadata_from_spec_dec(attn_metadata)

        if self._saved_kv_lens_cuda is not None:
            batch_size = self._saved_kv_lens_cuda.shape[0]
            attn_metadata.kv_lens_cuda[:batch_size].copy_(self._saved_kv_lens_cuda)
            self._saved_kv_lens_cuda = None

        if self._saved_packed_mask is not None:
            batch_size = self._saved_packed_mask.shape[0]
            attn_metadata.spec_decoding_packed_mask[:batch_size].copy_(self._saved_packed_mask)
            self._saved_packed_mask = None
        if self._saved_position_offsets is not None:
            attn_metadata.spec_decoding_position_offsets.copy_(self._saved_position_offsets)
            attn_metadata.spec_decoding_position_offsets_cpp = self._saved_position_offsets_cpp
            self._saved_position_offsets = None
            self._saved_position_offsets_cpp = None
        if self._saved_generation_lengths is not None:
            batch_size = self._saved_generation_lengths.shape[0]
            attn_metadata.spec_decoding_generation_lengths[:batch_size].copy_(
                self._saved_generation_lengths
            )
            self._saved_generation_lengths = None

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _apply_spec_metadata(self, attn_metadata, batch_size, query_len):
        """Set spec-dec gen lengths and refresh the C++ position-offset view."""
        attn_metadata.spec_decoding_generation_lengths[:batch_size] = query_len
        attn_metadata.update_position_offsets_for_cpp(query_len)

    def _refresh_blackwell_tree_mask_metadata(self, attn_metadata):
        if not getattr(attn_metadata, "use_spec_decoding", False):
            return
        if not getattr(attn_metadata, "is_spec_dec_dynamic_tree", False):
            return

        first_sparse = getattr(attn_metadata, "spec_bl_tree_first_sparse_mask_offset_kv", None)
        bl_tree_mask = getattr(attn_metadata, "spec_decoding_bl_tree_mask", None)
        if first_sparse is None and bl_tree_mask is None:
            return

        if bl_tree_mask is not None:
            bl_tree_mask.zero_()
        if first_sparse is not None:
            attn_metadata.update_blackwell_first_sparse_mask_offset()

    def _repack_mask_padded_to_packed(self, mask_buf, n_req, n_tok):
        """Compact padded masks into the flat prefix XQA expects."""
        buf_dim = mask_buf.shape[1]
        if n_tok >= buf_dim or n_req <= 1:
            return
        mask_width = math.ceil(n_tok / 32)
        total_elems = n_req * n_tok * mask_width
        scratch = self._mask_repack_buf[:total_elems].view(n_req, n_tok, mask_width)
        scratch.copy_(mask_buf[:n_req, :n_tok, :mask_width])
        flat = mask_buf.view(-1)
        flat[:total_elems] = scratch.view(-1)

    @nvtx_range("mtp_dyn._ensure_spec_tree_manager")
    def _ensure_spec_tree_manager(self, resource_manager):
        """Lazily bind spec_tree_manager and KV head metadata."""
        if self.spec_tree_manager is not None:
            return
        from ..pyexecutor.resource_manager import ResourceManagerType

        spec_rm = resource_manager.get_resource_manager(ResourceManagerType.SPEC_RESOURCE_MANAGER)
        assert spec_rm is not None and hasattr(spec_rm, "spec_tree_manager"), (
            "Dynamic tree mode requires spec_tree_manager in resource_manager"
        )
        self.spec_tree_manager = spec_rm.spec_tree_manager

        if self._kv_head_dim_bytes is None:
            cache_mgr = resource_manager.get_resource_manager(ResourceManagerType.KV_CACHE_MANAGER)
            if cache_mgr is not None and hasattr(cache_mgr, "head_dim"):
                from tensorrt_llm.bindings import DataType

                _dtype_bytes = {
                    DataType.HALF: 2,
                    DataType.BF16: 2,
                    DataType.FLOAT: 4,
                    DataType.FP8: 1,
                    DataType.INT8: 1,
                    DataType.NVFP4: 0.5,
                }
                self._kv_head_dim_bytes = int(
                    cache_mgr.head_dim * _dtype_bytes.get(cache_mgr.dtype, 0.5)
                )

    @nvtx_range("mtp_dyn.sample")
    def sample(
        self, logits: torch.Tensor, max_top_k: int, draft_model=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TopK sampling for dynamic tree; all-gather sharded TP logits."""
        mapping = (
            getattr(self.model_config, "mapping", None) if self.model_config is not None else None
        )
        if mapping is not None and mapping.tp_size > 1 and not mapping.enable_attention_dp:
            logits = allgather(logits, mapping, dim=-1)
            if draft_model is not None:
                vocab_size = draft_model.lm_head.num_embeddings
                logits = logits[..., :vocab_size]
        probs = torch.softmax(logits, dim=-1)
        topk_values, topk_indices = torch.topk(probs, k=max_top_k, dim=-1)
        return topk_indices, topk_values

    def update_draft_tokens_and_scores(
        self,
        cur_draft_idx,
        new_draft_tokens,
        new_draft_scores,
        previous_draft_scores,
        batch_size,
        attn_metadata=None,
    ):
        """Grow the tree and update history buffers."""
        if cur_draft_idx == 0:
            new_draft_scores = new_draft_scores.reshape(batch_size, self.K)
            new_draft_tokens_2d = new_draft_tokens.reshape(batch_size, self.K)
            self.draft_tokens_buffer[:batch_size, : self.K] = new_draft_tokens_2d
            self.history_draft_tokens_buffer[:batch_size, : self.K] = new_draft_tokens_2d
            self.history_score_buffer[:batch_size, : self.K] = new_draft_scores
            # Parent buffer: -1 for root, 0..K-1 for first layer.
            self.history_draft_tokens_parent_buffer[:batch_size, : self.K + 1] = (
                self._parent_init_arange
            )
            self.prepare_tree_mask_and_position_offset(cur_draft_idx, attn_metadata, None)
            return new_draft_scores

        (
            real_draft_tokens,
            topk_values,
            topk_indices,
            selected_parents,
            new_draft_tokens,
            new_draft_scores,
        ) = _select_topk_draft_tokens(
            new_draft_tokens, new_draft_scores, previous_draft_scores, self.K
        )

        num_tokens_previous_layer = cur_draft_idx * self.K
        num_tokens_current_layer = (cur_draft_idx + 1) * self.K
        self.draft_tokens_buffer[
            :batch_size, num_tokens_previous_layer:num_tokens_current_layer
        ] = real_draft_tokens

        write_start = self.K + (cur_draft_idx - 1) * self.K * self.K
        write_end = write_start + self.K * self.K
        self.history_draft_tokens_buffer[:batch_size, write_start:write_end] = new_draft_tokens
        self.history_score_buffer[:batch_size, write_start:write_end] = new_draft_scores

        self._last_selected_parents = selected_parents
        self.prepare_tree_mask_and_position_offset(cur_draft_idx, attn_metadata, selected_parents)

        if cur_draft_idx < self.max_draft_len - 1:
            next_layer_start = cur_draft_idx * self.K + 1
            next_layer_end = next_layer_start + self.K
            parents_relative_indices = topk_indices + self.K**2 * (cur_draft_idx - 1) + self.K
            self.history_draft_tokens_parent_buffer[
                :batch_size, next_layer_start:next_layer_end
            ] = parents_relative_indices
        return topk_values

    def resampling_final_draft_tokens(self, batch_size: int):
        """Reconstruct the final tree from history buffers."""
        return _resample_final_tokens(
            self.history_score_buffer[:batch_size, :],
            self.history_draft_tokens_buffer[:batch_size, :],
            self.max_total_draft_tokens,
        )

    def prepare_tree_mask_and_position_offset(
        self, cur_draft_idx, attn_metadata, selected_parents=None
    ):
        """Prepare mask and position offsets for the next draft layer."""
        if attn_metadata.spec_decoding_packed_mask is None:
            return
        spec_tree_manager = self.spec_tree_manager
        batch_size = attn_metadata.num_seqs
        num_tokens_current_layer = self.K * (cur_draft_idx + 1)
        num_tokens_previous_layer = self.K * cur_draft_idx
        packed_mask = attn_metadata.spec_decoding_packed_mask
        if cur_draft_idx == 0:
            spec_tree_manager.compute_spec_dec_packed_mask(
                self.tree_mask_init_buffer[:batch_size],
                packed_mask[:batch_size, :num_tokens_current_layer, :],
            )
            self.tree_mask_buffer[
                : batch_size * num_tokens_current_layer * num_tokens_current_layer
            ].copy_(self.tree_mask_init_buffer[:batch_size].view(-1))
            attn_metadata.spec_decoding_position_offsets.fill_(0)
            self._apply_spec_metadata(attn_metadata, batch_size, num_tokens_current_layer)
        else:
            num_parent_mask = batch_size * cur_draft_idx * self.K * cur_draft_idx * self.K
            parent_mask = self.tree_mask_buffer[:num_parent_mask].reshape(
                batch_size, cur_draft_idx * self.K, cur_draft_idx * self.K
            )

            prev_total = batch_size * num_tokens_previous_layer
            previous_position_offsets = attn_metadata.spec_decoding_position_offsets[
                :prev_total
            ].view(batch_size, num_tokens_previous_layer)

            current_mask, new_positions = _build_mask_and_position(
                parent_mask,
                selected_parents,
                self.tree_mask_init_buffer[:batch_size],
                previous_position_offsets,
                self.K,
            )

            spec_tree_manager.compute_spec_dec_packed_mask(
                current_mask, packed_mask[:batch_size, :num_tokens_current_layer, :]
            )
            self.tree_mask_buffer[
                : batch_size * num_tokens_current_layer * num_tokens_current_layer
            ].copy_(current_mask.reshape(-1))

            cur_total = batch_size * num_tokens_current_layer
            attn_metadata.spec_decoding_position_offsets[:cur_total] = new_positions.reshape(-1)
            self._apply_spec_metadata(attn_metadata, batch_size, num_tokens_current_layer)

        if self._needs_mask_repack:
            self._repack_mask_padded_to_packed(packed_mask, batch_size, num_tokens_current_layer)

    def update_hidden_states(
        self,
        cur_draft_idx,
        batch_size,
        step0_hs=None,
        hidden_states_to_save=None,
        selected_parents=None,
    ):
        """Manage growing-context hidden states for the MTP draft loop."""
        if cur_draft_idx == 0:
            hs_dim = step0_hs.shape[-1]
            self._hs_dim = hs_dim
            if self._hs_write_buffer is None or self._hs_write_buffer.shape[2] != hs_dim:
                self._hs_write_buffer = torch.zeros(
                    self._max_batch_size,
                    self.max_draft_len * self.K,
                    hs_dim,
                    device=step0_hs.device,
                    dtype=step0_hs.dtype,
                )
            if self._accumulated_hs is None or self._accumulated_hs.shape[2] != hs_dim:
                self._accumulated_hs = torch.zeros(
                    self._max_batch_size,
                    self.max_draft_len * self.K,
                    hs_dim,
                    device=step0_hs.device,
                    dtype=step0_hs.dtype,
                )
            # All K depth-0 tokens share step0_hs (the parent hidden state).
            self._accumulated_hs[:batch_size, : self.K] = step0_hs.unsqueeze(1).expand(
                -1, self.K, -1
            )
            self._step0_hs = step0_hs
        else:
            num_tokens_per_req = cur_draft_idx * self.K
            hs_to_save_reshaped = hidden_states_to_save.reshape(batch_size, num_tokens_per_req, -1)
            self._hs_write_buffer[:batch_size, :num_tokens_per_req] = hs_to_save_reshaped
            parent_offset = (cur_draft_idx - 1) * self.K
            self._hs_read_map[
                :batch_size, cur_draft_idx * self.K : (cur_draft_idx + 1) * self.K
            ] = parent_offset + selected_parents
            num_tokens_next = (cur_draft_idx + 1) * self.K
            read_idx = self._hs_read_map[:batch_size, self.K : num_tokens_next]
            hs_dim = self._hs_write_buffer.shape[2]
            self._accumulated_hs[:batch_size, self.K : num_tokens_next] = torch.gather(
                self._hs_write_buffer[:batch_size], 1, read_idx.unsqueeze(-1).expand(-1, -1, hs_dim)
            )

    # ------------------------------------------------------------------ #
    # Verification (greedy only)                                          #
    # ------------------------------------------------------------------ #
    @nvtx_range("mtp_dyn.sample_and_accept_draft_tokens")
    def sample_and_accept_draft_tokens(self, input_ids, logits, spec_metadata, attn_metadata):
        """Greedy verification of the previous dynamic tree."""
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        N = self.tokens_per_gen_step
        max_path_len = self._max_path_len

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # Reset output buffers.
        self._accepted_tokens_buf[:batch_size].zero_()
        accepted_tokens = self._accepted_tokens_buf[:batch_size, :max_path_len]
        self._num_accepted_tokens_buf[:batch_size].fill_(1)
        num_accepted_tokens = self._num_accepted_tokens_buf[:batch_size]
        self._accepted_draft_indices_tensor[:batch_size].fill_(-1)

        num_flat_tokens = logits.shape[0]
        torch.argmax(logits, dim=-1, out=self._target_tokens_buf[:num_flat_tokens])
        target_tokens = self._target_tokens_buf[:num_flat_tokens]

        # Context requests: accept the sampled golden token only.
        accepted_tokens[:num_contexts, 0].copy_(target_tokens[:num_contexts])

        if num_gens > 0:
            spec_tree_manager = self.spec_tree_manager
            target_predict = self._target_predict_buf[:num_gens]
            target_predict.copy_(target_tokens[num_contexts:].reshape(num_gens, N))

            # No prior tree exists on bootstrap/warmup; accept the golden token.
            if spec_tree_manager is None:
                num_accepted_tokens[num_contexts:batch_size] = 1
                accepted_tokens[num_contexts:batch_size, 0] = target_predict[:, 0]
                self._accepted_draft_indices_tensor[num_contexts:batch_size] = -1
                return accepted_tokens, num_accepted_tokens

            # candidates[:, 0] = golden token, candidates[:, 1:] = draft tokens.
            candidates = self._candidates_buf[:num_gens]
            candidates[:, 1:] = spec_metadata.draft_tokens.reshape(num_gens, N - 1)
            candidates[:, 0] = target_predict[:, 0]

            slot_storage = spec_tree_manager.slot_storage
            gen_slot_ids = slot_storage.all_ids_buf[num_contexts : num_contexts + num_gens]
            tree_valid = slot_storage.has_tree[gen_slot_ids]
            retrieve_packed = slot_storage.pack_retrieve_from_slots(gen_slot_ids, num_gens)

            accept_index, accept_token_num, accept_token = (
                self.tree_ops_converter.verify_dynamic_tree_greedy_out_packed(
                    candidates,
                    retrieve_packed,
                    target_predict,
                    num_gens,
                    self._max_path_len,
                    tree_valid=tree_valid,
                )
            )
            tree_valid_i = tree_valid[:num_gens]
            accepted_draft_count = torch.where(
                tree_valid_i,
                accept_token_num[:num_gens],
                torch.zeros_like(accept_token_num[:num_gens]),
            )
            num_accepted_tokens[num_contexts:batch_size] = (accepted_draft_count + 1).to(
                torch.int32
            )

            gen_accepted_tokens = accept_token[:num_gens].to(torch.int32)
            bootstrap_accepted_tokens = torch.zeros_like(gen_accepted_tokens)
            bootstrap_accepted_tokens[:, 0] = target_predict[:, 0]
            accepted_tokens[num_contexts:batch_size] = torch.where(
                tree_valid_i.unsqueeze(1), gen_accepted_tokens, bootstrap_accepted_tokens
            )
            # Convert root/padding index 0 to draft-node sentinel -1.
            gen_accepted_indices = (accept_index[:num_gens, 1:max_path_len] - 1).to(torch.int32)
            self._accepted_draft_indices_tensor[num_contexts:batch_size] = torch.where(
                tree_valid_i.unsqueeze(1),
                gen_accepted_indices,
                torch.full_like(gen_accepted_indices, -1),
            ).to(torch.int32)

        num_accepted_tokens = self._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, self.max_draft_len
        )
        return accepted_tokens, num_accepted_tokens

    def _accepted_leaf_intermediate_positions(self, num_accepted_tokens, num_contexts, num_gens):
        """Return each accepted leaf's position in the Mamba state buffer."""
        accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens].to(torch.int64)
        # Column of the deepest accepted draft node, clamped to >=0 for the
        # golden-only case (its value is ignored via the mask below).
        draft_idx = self._accepted_draft_indices_tensor[num_contexts : num_contexts + num_gens].to(
            torch.int64
        )
        last_col = (accepted - 2).clamp_(min=0, max=draft_idx.shape[1] - 1)
        leaf = torch.gather(draft_idx, 1, last_col.unsqueeze(1)).squeeze(1) + 1
        # Golden-only requests (num_accepted == 1) take the root at position 0.
        return torch.where(accepted > 1, leaf, torch.zeros_like(leaf))

    @nvtx_range("mtp_dyn._relocate_kv_eagerly")
    def _relocate_kv_eagerly(self, attn_metadata, batch_size):
        """Move accepted draft KV from tree positions to the linear prefix."""
        cache_mgr = getattr(attn_metadata, "kv_cache_manager", None)
        if cache_mgr is None or self._kv_head_dim_bytes is None:
            return
        if not hasattr(cache_mgr, "num_kv_heads_per_layer"):
            return

        # Mamba layers have zero KV heads; relocate attention-layer KV only.
        kv_heads = cache_mgr.num_kv_heads_per_layer
        attn_heads = set(h for h in kv_heads if h > 0)
        assert len(attn_heads) == 1, (
            "update_kv_cache_draft_token_location_2d requires uniform "
            f"num_kv_heads across attention layers, got {list(kv_heads)}"
        )
        attn_num_heads = attn_heads.pop()
        attn_layer_offsets = [i for i, h in enumerate(kv_heads) if h > 0]
        attn_num_layers = len(attn_layer_offsets)

        # Resolve the attention KV pool used by the relocation op.
        pool_mapping = getattr(cache_mgr, "kv_cache_pool_mapping", None)
        if pool_mapping is not None:
            attn_pool_indices = set(int(pool_mapping[off][0]) for off in attn_layer_offsets)
            assert len(attn_pool_indices) == 1, (
                "update_kv_cache_draft_token_location_2d requires all attention "
                f"layers in one KV pool, got pools {sorted(attn_pool_indices)}"
            )
            attn_pool_idx = attn_pool_indices.pop()
        else:
            attn_pool_idx = 0

        pool_pointers = cache_mgr.kv_cache_pool_pointers[attn_pool_idx]
        block_offsets = attn_metadata.kv_cache_block_offsets[attn_pool_idx]

        torch.ops.tensorrt_llm.update_kv_cache_draft_token_location_2d(
            self._accepted_draft_indices_tensor[:batch_size],
            self._num_accepted_tokens_buf[:batch_size],
            attn_metadata.kv_lens_cuda[:batch_size],
            True,
            attn_num_layers,
            attn_num_heads,
            self._kv_head_dim_bytes,
            cache_mgr.max_total_draft_tokens,
            cache_mgr.max_attention_window_vec[0],
            pool_pointers,
            block_offsets,
            cache_mgr.max_blocks_per_seq,
            cache_mgr.tokens_per_block,
            None,
        )

    # ------------------------------------------------------------------ #
    # Top-level forward                                                   #
    # ------------------------------------------------------------------ #
    @nvtx_range("mtp_dyn.forward")
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
        """Run verify, cache promotion, and next-tree drafting."""
        if resource_manager is not None:
            self._ensure_spec_tree_manager(resource_manager)

        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts
        raw_logits = logits

        self._execute_guided_decoder_if_present(logits)

        # (a) Verify previous tree (greedy). Also relocates accepted KV.
        accepted_tokens, num_accepted_tokens = self.sample_and_accept_draft_tokens(
            input_ids, logits, spec_metadata, attn_metadata
        )
        if num_gens > 0:
            self._relocate_kv_eagerly(attn_metadata, batch_size)

        # Dynamic-tree Mamba states are stored by tree-node position.
        if self._is_mamba_hybrid_cache is None:
            self._is_mamba_hybrid_cache = isinstance(
                attn_metadata.kv_cache_manager, MambaHybridCacheManager
            )
        if num_gens > 0 and self._is_mamba_hybrid_cache:
            accepted_leaf_positions = self._accepted_leaf_intermediate_positions(
                num_accepted_tokens, num_contexts, num_gens
            )
            attn_metadata.kv_cache_manager.update_mamba_states(
                attn_metadata=attn_metadata,
                num_accepted_tokens=num_accepted_tokens,
                state_indices=attn_metadata.mamba_metadata.state_indices,
                accepted_leaf_positions=accepted_leaf_positions,
            )

        # Save attn/spec metadata before the draft loop mutates it.
        original_all_rank_num_tokens = attn_metadata.all_rank_num_tokens
        original_force_prepare_spec_dec_tree_mask = attn_metadata.force_prepare_spec_dec_tree_mask
        self._prepare_attn_metadata_for_spec_dec(attn_metadata)
        attn_metadata.force_prepare_spec_dec_tree_mask = True

        # (c) Run the MTP draft tree loop -> build + store the next tree.
        draft_kv_cache_manager = self.get_draft_kv_cache_manager(resource_manager)
        next_draft_tokens = self._forward_draft_loop(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            attn_metadata=attn_metadata,
            spec_metadata=spec_metadata,
            draft_model=draft_model,
            draft_kv_cache_manager=draft_kv_cache_manager,
            num_contexts=num_contexts,
            num_gens=num_gens,
            batch_size=batch_size,
        )

        # Restore attn metadata to support cuda graph.
        self._restore_attn_metadata_from_spec_dec(attn_metadata)
        attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens
        attn_metadata.force_prepare_spec_dec_tree_mask = original_force_prepare_spec_dec_tree_mask
        attn_metadata.use_spec_decoding = True

        # (d) Prepare next_new_tokens for overlap scheduler.
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
            "accepted_draft_tokens_indices": self._accepted_draft_indices_tensor[:batch_size],
        }

    # ------------------------------------------------------------------ #
    # Step-0 drafter-input repack (dynamic tree)                          #
    # ------------------------------------------------------------------ #
    @nvtx_range("mtp_dyn._prepare_step0_drafter_inputs")
    def _prepare_step0_drafter_inputs(
        self,
        input_ids,
        position_ids,
        last_tokens_idx,
        hidden_states,
        accepted_tokens,
        attn_metadata,
    ):
        """Repack step-0 drafter inputs to accepted-path layout."""
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts
        num_ctx_tokens = attn_metadata.num_ctx_tokens

        # Match MTPEagleWorker context input repack.
        input_ids_ctx = self._prepare_context_input_ids(
            input_ids, num_ctx_tokens, last_tokens_idx, accepted_tokens, num_contexts
        )

        if num_gens > 0:
            max_path_len = self._max_path_len
            num_gen_tokens = num_gens * max_path_len

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

            # Accepted path includes the golden token at column 0.
            accept_token = accepted_tokens[num_contexts:batch_size]

            BLOCK_H = triton.next_power_of_2(hidden_dim)
            _gather_repack_step0_kernel[(num_gens * max_path_len,)](
                hidden_states,
                accept_token,
                position_ids,
                self._accepted_draft_indices_tensor[num_contexts:batch_size],
                self._num_accepted_tokens_buf,
                self._step0_hidden_states_buf,
                self._step0_input_ids_buf,
                self._step0_position_ids_buf,
                self._gather_ids_buf,
                num_ctx_tokens,
                num_contexts,
                self.tokens_per_gen_step,
                max_path_len,
                self.max_draft_len,
                hidden_dim,
                num_ctx_tokens,  # gather_id references combined [ctx|gen] tensor
                BLOCK_H=BLOCK_H,
            )

            input_ids = torch.cat(
                [input_ids_ctx, self._step0_input_ids_buf[:num_gen_tokens]], dim=0
            )
            position_ids = torch.cat(
                [position_ids[:num_ctx_tokens], self._step0_position_ids_buf[:num_gen_tokens]],
                dim=0,
            )
            hidden_states = torch.cat(
                [hidden_states[:num_ctx_tokens], self._step0_hidden_states_buf[:num_gen_tokens]],
                dim=0,
            )

            attn_metadata._seq_lens[num_contexts:batch_size].fill_(max_path_len)
            attn_metadata._seq_lens_cuda[num_contexts:batch_size].fill_(max_path_len)
            attn_metadata.on_update()
        else:
            # Context-only (warmup): no gen tokens to repack.
            input_ids = input_ids_ctx

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
        }

    # ------------------------------------------------------------------ #
    # MTP draft tree loop                                                 #
    # ------------------------------------------------------------------ #
    def _forward_draft_loop(
        self,
        input_ids,
        position_ids,
        hidden_states,
        accepted_tokens,
        num_accepted_tokens,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager,
        num_contexts,
        num_gens,
        batch_size,
    ):
        """Draft the next dynamic tree with growing context."""
        spec_tree_manager = self.spec_tree_manager

        assert batch_size <= self._max_batch_size, (
            f"batch_size {batch_size} exceeds pre-allocated max_batch_size {self._max_batch_size}"
        )

        # Step 0: run MTP over accepted-path rows.
        position_ids, last_tokens_idx = self.prepare_position_ids_and_last_tokens(
            position_ids, attn_metadata.seq_lens_cuda
        )
        inputs = self._prepare_step0_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            last_tokens_idx=last_tokens_idx,
            hidden_states=hidden_states,
            accepted_tokens=accepted_tokens,
            attn_metadata=attn_metadata,
        )

        # Reset verify-time tree metadata to accepted-path width.
        num_step0_tokens = self._max_path_len
        if attn_metadata.spec_decoding_generation_lengths is not None:
            total = num_gens * num_step0_tokens
            dst = attn_metadata.spec_decoding_position_offsets[:total].view(
                num_gens, num_step0_tokens
            )
            dst.copy_(self._causal_offs[:num_step0_tokens].unsqueeze(0).expand(num_gens, -1))
            self._apply_spec_metadata(attn_metadata, num_gens, num_step0_tokens)
            packed_mask = attn_metadata.spec_decoding_packed_mask
            packed_mask[:num_gens].zero_()
            packed_mask[:num_gens, :num_step0_tokens, 0] = self._step0_causal_mask[
                :num_step0_tokens
            ]
            if self._needs_mask_repack:
                self._repack_mask_padded_to_packed(packed_mask, num_gens, num_step0_tokens)
        attn_metadata.use_spec_decoding = num_gens > 0
        if num_gens > 0 and hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= self._kv_correction
        self._refresh_blackwell_tree_mask_metadata(attn_metadata)
        if spec_metadata.all_rank_num_tokens is not None:
            # Keep attention/MoE token counts aligned with step-0 repack.
            attn_metadata.all_rank_num_tokens = spec_metadata.all_rank_num_tokens

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            hidden_states = draft_model.mtp_layers[0](
                embed_tokens=draft_model.embed_tokens,
                all_rank_num_tokens=spec_metadata.all_rank_num_tokens,
                **inputs,
            )

            # Gather each request's root hidden state for depth-0 expansion.
            self._gather_ids_buf[:num_contexts].copy_(last_tokens_idx[:num_contexts])
            gather_ids = self._gather_ids_buf[:batch_size]

            step0_hs = hidden_states[gather_ids]
            logits = draft_model.mtp_layers[0].shared_head(
                step0_hs, draft_model.lm_head, attn_metadata, True
            )

            new_draft_tokens, new_draft_scores = self.sample(
                logits, self.K, draft_model=draft_model
            )
            previous_draft_scores = self.update_draft_tokens_and_scores(
                cur_draft_idx=0,
                new_draft_tokens=new_draft_tokens,
                new_draft_scores=new_draft_scores,
                previous_draft_scores=None,
                batch_size=batch_size,
                attn_metadata=attn_metadata,
            )
            self.update_hidden_states(cur_draft_idx=0, batch_size=batch_size, step0_hs=step0_hs)
            self._prepare_draft_layer_metadata(
                0,
                attn_metadata,
                batch_size,
                gather_ids,
                num_contexts,
                num_gens,
                num_accepted_tokens,
                inputs,
            )

            # Subsequent layers grow the tree.
            for layer_idx in range(1, self.max_draft_len):
                num_tokens_per_req = layer_idx * self.K
                num_infer_tokens = batch_size * num_tokens_per_req
                subseq_all_rank_num_tokens = None
                if spec_metadata.all_rank_num_seqs is not None:
                    # Token counts scale with the current tree width.
                    subseq_all_rank_num_tokens = [
                        n * num_tokens_per_req for n in spec_metadata.all_rank_num_seqs
                    ]
                    attn_metadata.all_rank_num_tokens = subseq_all_rank_num_tokens

                inp_hs = self._accumulated_hs[:batch_size, :num_tokens_per_req, :].reshape(
                    num_infer_tokens, -1
                )
                inp_ids = self.draft_tokens_buffer[:batch_size, :num_tokens_per_req].reshape(-1)
                inp_pos = self.position_ids_buffer[:batch_size, :num_tokens_per_req].reshape(-1)
                layer_inputs = {
                    "input_ids": inp_ids,
                    "position_ids": inp_pos,
                    "hidden_states": inp_hs,
                    "attn_metadata": attn_metadata,
                }
                hidden_states = draft_model.mtp_layers[0](
                    embed_tokens=draft_model.embed_tokens,
                    all_rank_num_tokens=subseq_all_rank_num_tokens
                    or spec_metadata.subseq_all_rank_num_tokens,
                    **layer_inputs,
                )

                # Take the last K hidden states per request (the new leaves).
                hs_reshaped = hidden_states.reshape(batch_size, num_tokens_per_req, -1)
                selected_hs = hs_reshaped[:, -self.K :, :].reshape(batch_size * self.K, -1)
                logits = draft_model.mtp_layers[0].shared_head(
                    selected_hs, draft_model.lm_head, attn_metadata, True
                )

                new_draft_tokens, new_draft_scores = self.sample(
                    logits, self.K, draft_model=draft_model
                )
                new_draft_tokens = new_draft_tokens.reshape(batch_size, self.K, self.K)
                new_draft_scores = new_draft_scores.reshape(batch_size, self.K, self.K)

                previous_draft_scores = self.update_draft_tokens_and_scores(
                    cur_draft_idx=layer_idx,
                    new_draft_tokens=new_draft_tokens,
                    new_draft_scores=new_draft_scores,
                    previous_draft_scores=previous_draft_scores,
                    batch_size=batch_size,
                    attn_metadata=attn_metadata,
                )
                self.update_hidden_states(
                    cur_draft_idx=layer_idx,
                    batch_size=batch_size,
                    hidden_states_to_save=hidden_states,
                    selected_parents=self._last_selected_parents,
                )
                self._prepare_draft_layer_metadata(layer_idx, attn_metadata, batch_size)

        # Resample the final tree and build it into slot_storage.
        real_draft_tokens, topk_score_indices = self.resampling_final_draft_tokens(batch_size)

        if spec_tree_manager is not None and num_gens > 0:
            self.tree_ops_converter.build_dynamic_tree(
                history_draft_tokens_parent_buffer=self.history_draft_tokens_parent_buffer[
                    num_contexts:batch_size
                ],
                topk_score_indices=topk_score_indices[num_contexts:],
                tree_mask=spec_tree_manager.spec_dec_packed_mask[:num_gens],
                positions=spec_tree_manager.spec_dec_position_offsets[:num_gens],
                retrieve_index=spec_tree_manager.retrieve_index[:num_gens],
                retrieve_next_token=spec_tree_manager.retrieve_next_token[:num_gens],
                retrieve_next_sibling=spec_tree_manager.retrieve_next_sibling[:num_gens],
                use_packed_mask=True,
            )
            slot_storage = spec_tree_manager.slot_storage
            gen_slots = slot_storage.all_ids_buf[num_contexts:batch_size]
            spec_tree_manager.scatter_to_slot_storage(slot_storage, gen_slots, num_gens)

        return real_draft_tokens

    def _prepare_draft_layer_metadata(
        self,
        cur_draft_idx,
        attn_metadata,
        batch_size,
        gather_ids=None,
        num_contexts=0,
        num_gens=0,
        num_accepted_tokens=None,
        inputs=None,
    ):
        """Set attn_metadata seq_lens/kv_lens for the next draft layer."""
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
                # Rewind only unaccepted verify tokens; draft KV is added later.
                if num_gens > 0:
                    attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= (
                        self._max_path_len
                    ) - num_accepted_tokens[num_contexts:batch_size]
                attn_metadata.kv_lens_cuda[:batch_size] += self.K
            attn_metadata.use_spec_decoding = True
            self._refresh_blackwell_tree_mask_metadata(attn_metadata)
        else:
            num_tokens_previous_layer = cur_draft_idx * self.K
            num_tokens_current_layer = self.K * (cur_draft_idx + 1)
            prev_pos = self.position_ids_buffer[:batch_size, :num_tokens_previous_layer]
            self.position_ids_buffer[
                :batch_size, num_tokens_previous_layer:num_tokens_current_layer
            ] = prev_pos[:, -self.K :] + 1
            attn_metadata._seq_lens[:batch_size].fill_(num_tokens_current_layer)
            attn_metadata._seq_lens_cuda[:batch_size].fill_(num_tokens_current_layer)
            attn_metadata.on_update()
            if hasattr(attn_metadata, "kv_lens_cuda"):
                attn_metadata.kv_lens_cuda[:batch_size] += self.K
            self._refresh_blackwell_tree_mask_metadata(attn_metadata)


class MTPEagleDynamicTreeResourceManager(BaseResourceManager):
    """Resource manager for MTP dynamic-tree mode."""

    hidden_states: Optional[torch.Tensor] = None

    def __init__(
        self,
        config: "MTPDecodingConfig",
        dtype: torch.dtype,
        hidden_size: int,
        max_num_requests: int,
        sa_manager=None,
    ):
        from .spec_tree_manager import SpecTreeManager

        self.max_num_requests = max_num_requests
        self.spec_tree_manager = SpecTreeManager(
            max_num_requests=max_num_requests,
            use_dynamic_tree=True,
            max_draft_len=config.max_draft_len,
            max_total_draft_tokens=config.tokens_per_gen_step - 1,
            eagle_choices=None,
            dynamic_tree_max_topK=config.dynamic_tree_max_topK,
        )
        # MTP hidden-state slot pools (needed by MTPEagleWorker drafter inputs).
        self._mtp_hidden_states_manager = MTPHiddenStatesManager(
            config, dtype, hidden_size, max_num_requests, sa_manager=sa_manager
        )

    # Expose the MTPHiddenStatesManager surface MTPSpecMetadata expects.
    @property
    def slot_manager(self):
        return self._mtp_hidden_states_manager.slot_manager

    @property
    def mtp_past_hidden_states_pool(self):
        return self._mtp_hidden_states_manager.mtp_past_hidden_states_pool

    @property
    def mtp_past_tokens_pool(self):
        return self._mtp_hidden_states_manager.mtp_past_tokens_pool

    @property
    def sa_manager(self):
        return self._mtp_hidden_states_manager.sa_manager

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        self._mtp_hidden_states_manager.prepare_resources(scheduled_batch)

    def update_resources(self, scheduled_batch: ScheduledRequests):
        self._mtp_hidden_states_manager.update_resources(scheduled_batch)

    def free_resources(self, request: LlmRequest):
        # Clear tree validity for the freed slot, then free the MTP slot.
        if request.py_seq_slot is not None:
            self.spec_tree_manager.slot_storage.mark_invalid(request.py_seq_slot)
        self._mtp_hidden_states_manager.free_resources(request)

    def add_dummy_requests(self, request_ids: List[int]):
        # Dummies still need MTP hidden-state slots.
        self._mtp_hidden_states_manager.add_dummy_requests(request_ids)

    def shutdown(self):
        self._mtp_hidden_states_manager.shutdown()

    def get_max_resource_count(self) -> int:
        return self.max_num_requests

    def get_needed_resource_to_completion(self, request: LlmRequest):
        return 0
