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

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import get_sm_version, nvtx_range

from ..attention_backend import AttentionMetadata
from .eagle3 import Eagle3OneModelWorker

if TYPE_CHECKING:
    from ...llmapi.llm_args import EagleDecodingConfig


@torch.compile(options={"max-autotune": True})
def _sample_softmax_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax+topk for draft token sampling."""
    last_p = torch.softmax(logits, dim=-1)
    topk_values, topk_indices = torch.topk(last_p, k=k, dim=-1)
    return topk_indices, topk_values


@torch.compile(options={"max-autotune": True})
def _select_topk_draft_tokens(
    new_draft_tokens: torch.Tensor,
    new_draft_scores: torch.Tensor,
    previous_draft_scores: torch.Tensor,
    K: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused score accumulation → topk → gather for draft layer selection.

    Returns (real_tokens, topk_values, topk_indices, parents,
             all_tokens_flat, all_scores_flat) where the last two are the full
             K*K reshaped tensors needed for history buffer writes.
    """
    all_tokens_flat = new_draft_tokens.reshape(-1, K * K)
    all_scores_flat = (new_draft_scores * previous_draft_scores.unsqueeze(2)).reshape(-1, K * K)
    topk_values, topk_indices = torch.topk(all_scores_flat, k=K, dim=-1)
    real_tokens = torch.gather(all_tokens_flat, dim=1, index=topk_indices)
    parents = topk_indices // K
    return real_tokens, topk_values, topk_indices, parents, all_tokens_flat, all_scores_flat


@torch.compile(options={"max-autotune": True})
def _resample_final_tokens(
    history_scores: torch.Tensor,
    history_tokens: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused topk → sort → gather for final tree token resampling."""
    topk_score_indices = torch.topk(history_scores, k=k, dim=-1).indices
    topk_score_indices = torch.sort(topk_score_indices).values
    real_draft_tokens = torch.gather(history_tokens, dim=1, index=topk_score_indices)
    return real_draft_tokens, topk_score_indices


@triton.jit
def _gather_repack_step0_kernel(
    hidden_states_ptr,
    accept_token_ptr,
    position_ids_ptr,
    accepted_indices_ptr,
    num_accepted_ptr,
    out_hs_ptr,
    out_ids_ptr,
    out_pos_ptr,
    out_gather_ids_ptr,
    num_ctx_tokens,
    num_contexts,
    tokens_per_gen_step,
    max_path_len,
    max_draft_len,
    hidden_dim,
    gather_id_offset,
    BLOCK_H: tl.constexpr,
):
    """Fused gather+repack for generation requests in step 0.

    Each program handles one (gen_request, path_position) pair:
    - Gathers hidden_states from tree-topology to contiguous layout
    - Copies input_id from accept_token
    - Computes position_id from base_position + path_offset
    - Computes gather_id for the last accepted position
    """
    pid = tl.program_id(0)
    gen_idx = pid // max_path_len
    path_idx = pid % max_path_len

    # Source position in tree-topology layout
    gen_start = num_ctx_tokens + gen_idx * tokens_per_gen_step
    # tl.where evaluates both branches, so clamp index to avoid OOB when path_idx==0
    safe_idx = tl.maximum(path_idx - 1, 0)
    raw_val = tl.load(accepted_indices_ptr + gen_idx * max_draft_len + safe_idx)
    tree_pos = tl.where(path_idx == 0, 0, raw_val.to(tl.int64) + 1)
    src_row = gen_start + tree_pos

    dst_row = gen_idx * max_path_len + path_idx

    # 1) Gather hidden_states: src[src_row, :] → out[dst_row, :]
    for h in range(0, hidden_dim, BLOCK_H):
        offsets = h + tl.arange(0, BLOCK_H)
        mask = offsets < hidden_dim
        vals = tl.load(hidden_states_ptr + src_row * hidden_dim + offsets, mask=mask)
        tl.store(out_hs_ptr + dst_row * hidden_dim + offsets, vals, mask=mask)

    # 2) Copy input_id
    token = tl.load(accept_token_ptr + gen_idx * max_path_len + path_idx)
    tl.store(out_ids_ptr + dst_row, token)

    # 3) Compute position_id: base_pos + path_idx
    base_pos = tl.load(position_ids_ptr + gen_start)
    tl.store(out_pos_ptr + dst_row, base_pos + path_idx)

    # 4) Compute gather_id (only for last accepted token of each gen request)
    #    gather_id references the FINAL combined [ctx | gen] tensor
    n_acc = tl.load(num_accepted_ptr + num_contexts + gen_idx).to(tl.int64)
    if path_idx == 0:
        gather_id = gather_id_offset + gen_idx * max_path_len + n_acc - 1
        tl.store(out_gather_ids_ptr + num_contexts + gen_idx, gather_id)


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
        self.tokens_per_gen_step = spec_config.tokens_per_gen_step
        if spec_config.max_batch_size is None:
            raise ValueError(
                "Eagle3OneModelDynamicTreeWorker requires max_batch_size to be set "
                "on Eagle3DecodingConfig when use_dynamic_tree=True."
            )
        self._max_batch_size = spec_config.max_batch_size

        K = self.K
        max_draft_len = spec_config.max_draft_len
        max_batch_size = self._max_batch_size
        loop_max_tokens = K * max_draft_len  # draft loop working size

        # Pre-allocated 2D buffers
        self.draft_tokens_buffer = torch.zeros(
            max_batch_size, loop_max_tokens, dtype=torch.int64, device="cuda"
        )
        self.position_ids_buffer = torch.zeros(
            max_batch_size, loop_max_tokens, dtype=torch.int64, device="cuda"
        )
        self.history_draft_tokens_buffer = torch.zeros(
            (max_batch_size, (K + K * K * (max_draft_len - 1))), dtype=torch.int64, device="cuda"
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
            (max_batch_size * loop_max_tokens * loop_max_tokens),
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

        self._accept_token = torch.zeros(
            max_batch_size, max_draft_len + 1, dtype=torch.int64, device="cuda"
        )
        self._last_selected_parents = None
        self._kv_head_dim_bytes = None
        self._max_path_len = max_draft_len + 1
        self._kv_correction = self.tokens_per_gen_step - self._max_path_len

        # Hidden state management buffers (lazily initialized in update_hidden_states)
        self._hs_write_buffer = None
        self._hs_read_map = torch.zeros(
            max_batch_size, max_draft_len * K, dtype=torch.long, device="cuda"
        )
        self._accumulated_hs = None
        self._step0_hs = None
        self._hs_dim = None

        # Step 0 buffers
        self._step0_causal_mask = torch.tensor(
            [(1 << (t + 1)) - 1 for t in range(self._max_path_len)],
            dtype=torch.int32,
            device="cuda",
        )
        self._arange_max_batch = torch.arange(max_batch_size, device="cuda")
        self._causal_offs = torch.arange(self._max_path_len, device="cuda", dtype=torch.int32)
        self._parent_init_arange = torch.arange(-1, K, device="cuda", dtype=torch.int32)

        # Verification buffers
        tokens_per_gen_step = self.tokens_per_gen_step
        self._accepted_tokens_buf = torch.zeros(
            max_batch_size, self._max_path_len, dtype=torch.int32, device="cuda"
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
            loop_max_tokens,
            loop_max_tokens,
            dtype=torch.int32,
            device="cuda",
        )
        self._new_pos_offset_buf = torch.zeros(
            max_batch_size, loop_max_tokens, dtype=torch.int32, device="cuda"
        )

        # Mask repack scratch (graph-safe; avoids .contiguous() in the draft loop).
        buf_dim = max(self.max_total_draft_tokens + 1, K * max_draft_len)
        mask_width = (buf_dim + 31) // 32
        self._mask_repack_buf = torch.zeros(
            max_batch_size * buf_dim * mask_width, dtype=torch.int32, device="cuda"
        )

        # sm≥100 (except 120/121): prepareCustomMask keeps padded 3D; no 1D repack.
        sm = get_sm_version()
        self._needs_mask_repack = sm < 100 or sm in (120, 121)

    def _repack_mask_padded_to_packed(self, mask_buf, n_req, n_tok):
        """XQA indexes mask flat via cuQSeqLens; padded [n_req, buf_dim, maskW] has
        batch stride buf_dim*maskW, so when n_tok < buf_dim and n_req > 1 packed
        reads are wrong. Copy [:n_req,:n_tok] through scratch into flat prefix."""
        buf_dim = mask_buf.shape[1]
        if n_tok >= buf_dim or n_req <= 1:
            return
        mask_width = mask_buf.shape[2]
        total_elems = n_req * n_tok * mask_width
        scratch = self._mask_repack_buf[:total_elems].view(n_req, n_tok, mask_width)
        scratch.copy_(mask_buf[:n_req, :n_tok, :])
        flat = mask_buf.view(-1)
        flat[:total_elems] = scratch.view(-1)

    @nvtx_range("eagle3_dyn._ensure_spec_tree_manager")
    def _ensure_spec_tree_manager(self, resource_manager):
        """Lazily initialize spec_tree_manager and KV head metadata."""
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
            if cache_mgr is not None:
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

    @nvtx_range("eagle3_dyn.forward")
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
        # Initialize spec_tree_manager before super().forward() which calls
        # _forward_draft_loop needing spec_tree_manager.
        if resource_manager is not None:
            self._ensure_spec_tree_manager(resource_manager)
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

    @nvtx_range("eagle3_dyn._relocate_kv_eagerly")
    def _relocate_kv_eagerly(self, attn_metadata, batch_size):
        """Move accepted draft tokens' KV from tree to linear positions.

        Called inside sample_and_accept_draft_tokens (captured in CUDA graph).
        All tensor arguments are pre-allocated so their device addresses are
        stable across CUDA graph capture and replay.

        Must use attn_metadata.kv_cache_manager (the target model's live KV
        cache), as it may differ from ResourceManagerType.KV_CACHE_MANAGER in
        one-model spec decoding configurations.
        """
        cache_mgr = getattr(attn_metadata, "kv_cache_manager", None)
        if cache_mgr is None:
            return

        assert len(set(cache_mgr.num_kv_heads_per_layer)) == 1, (
            "update_kv_cache_draft_token_location_2d requires uniform num_kv_heads across all layers, "
            f"but got {cache_mgr.num_kv_heads_per_layer}"
        )
        torch.ops.tensorrt_llm.update_kv_cache_draft_token_location_2d(
            self._accepted_draft_indices_tensor[:batch_size],
            self._num_accepted_tokens_buf[:batch_size],
            attn_metadata.kv_lens_cuda[:batch_size],
            True,
            cache_mgr.num_layers,
            # Use TP-sharded num_kv_heads (per-rank) instead of the unsharded
            # total so the C++ kernel computes correct strides and grid dims.
            cache_mgr.num_kv_heads_per_layer[0],
            self._kv_head_dim_bytes,
            cache_mgr.max_total_draft_tokens,
            cache_mgr.max_attention_window_vec[0],
            cache_mgr.kv_cache_pool_pointers,
            attn_metadata.kv_cache_block_offsets,
            cache_mgr.max_blocks_per_seq,
            cache_mgr.tokens_per_block,
            None,
        )

    @nvtx_range("eagle3_dyn.sample_and_accept_draft_tokens")
    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        """Override to handle dynamic tree verification."""
        batch_size = attn_metadata.num_seqs
        num_contexts = attn_metadata.num_contexts
        num_gens = batch_size - num_contexts

        accepted_tokens, num_accepted_tokens = self._sample_and_accept_dynamic_tree(
            logits, attn_metadata, spec_metadata, batch_size, num_contexts, num_gens
        )
        if num_gens > 0:
            self._relocate_kv_eagerly(attn_metadata, batch_size)
        return accepted_tokens, num_accepted_tokens

    @nvtx_range("eagle3_dyn.prepare_1st_drafter_inputs")
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
        Uses a fused Triton kernel for gen requests to minimize kernel launches.
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

            # Gen tokens: fused Triton kernel writes into gen-only buffers
            # (buffers sized for max_batch_size * tokens_per_gen_step)
            BLOCK_H = triton.next_power_of_2(hidden_dim)
            _gather_repack_step0_kernel[(num_gens * max_path_len,)](
                hidden_states,
                self._accept_token[:num_gens],
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

            # Concat context + gen
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
            # Context-only (warmup). No gen tokens.
            input_ids = input_ids_ctx

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "hidden_states": hidden_states,
            "attn_metadata": attn_metadata,
            "spec_metadata": spec_metadata,
        }

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
        spec_tree_manager = self.spec_tree_manager

        assert batch_size <= self._max_batch_size, (
            f"batch_size {batch_size} exceeds pre-allocated max_batch_size {self._max_batch_size}"
        )

        # === Step 0: Initial forward ===
        num_step0_tokens = self._max_path_len

        # Triton kernel already wrote gen gather_ids into _gather_ids_buf;
        # just prepend context gather_ids.
        self._gather_ids_buf[:num_contexts].copy_(spec_metadata.gather_ids[:num_contexts])
        gather_ids = self._gather_ids_buf[:batch_size]

        if original_all_rank_num_tokens is not None:
            attn_metadata.all_rank_num_tokens = original_all_rank_num_tokens

        # Step-0 causal spec-dec (None in prefill-only warmup).
        if attn_metadata.spec_decoding_generation_lengths is not None:
            # Gen kernel uses base data_ptr (no num_contexts offset); fill rows [:num_gens].
            attn_metadata.spec_decoding_generation_lengths[:num_gens].fill_(num_step0_tokens)

            # Position stride num_step0_tokens matches C++ generation_input_length.
            pos_2d = attn_metadata.spec_decoding_position_offsets[
                : num_gens * num_step0_tokens
            ].view(num_gens, num_step0_tokens)
            pos_2d[:] = self._causal_offs[:num_step0_tokens]

            attn_metadata.spec_decoding_packed_mask[:num_gens].fill_(0)
            attn_metadata.spec_decoding_packed_mask[:num_gens, :num_step0_tokens, 0] = (
                self._step0_causal_mask[:num_step0_tokens]
            )
            # Packed flat for cuQSeqLens when _needs_mask_repack; else keep padded layout.
            if self._needs_mask_repack:
                self._repack_mask_padded_to_packed(
                    attn_metadata.spec_decoding_packed_mask, num_gens, num_step0_tokens
                )

        attn_metadata.use_spec_decoding = num_gens > 0

        # KV correction: target processed tokens_per_gen_step per request,
        # but step 0 should only attend to max_draft_len + 1 accepted-path tokens.
        if num_gens > 0 and hasattr(attn_metadata, "kv_lens_cuda"):
            attn_metadata.kv_lens_cuda[num_contexts:batch_size] -= self._kv_correction

        with self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager):
            hidden_states, hidden_states_to_save = draft_model.model(**inputs)

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
                batch_size=batch_size,
                attn_metadata=attn_metadata,
            )

            self.update_hidden_states(
                cur_draft_idx=0,
                batch_size=batch_size,
                step0_hs=step0_hs,
            )

            self.prepare_for_generation(
                0,
                attn_metadata,
                batch_size,
                inputs=inputs,
                gather_ids=gather_ids,
                num_contexts=num_contexts,
                num_gens=num_gens,
                num_accepted_tokens=num_accepted_tokens,
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
                    batch_size=batch_size,
                    attn_metadata=attn_metadata,
                )

                self.update_hidden_states(
                    cur_draft_idx=layer_idx,
                    batch_size=batch_size,
                    hidden_states_to_save=hidden_states_to_save,
                    selected_parents=self._last_selected_parents,
                )

                self.prepare_for_generation(layer_idx, attn_metadata, batch_size)

        # Resample final tokens and build tree
        real_draft_tokens, topk_score_indices = self.resampling_final_draft_tokens(batch_size)

        if spec_tree_manager is not None:
            # Gen-only; spec-dec batch is [:num_gens] (not num_contexts-offset).
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

            # Scatter only gen trees to stable slot storage
            gen_slots = spec_tree_manager._all_slot_ids_buf[num_contexts:batch_size]
            spec_tree_manager.scatter_trees_to_slots(gen_slots, num_gens)
            spec_tree_manager.mark_tree_valid(gen_slots, num_gens)

        return real_draft_tokens.to(torch.int32)

    def _sample_and_accept_dynamic_tree(
        self, logits, attn_metadata, spec_metadata, batch_size, num_contexts, num_gens
    ):
        """Dynamic tree verification using CUDA kernel."""
        N = self.tokens_per_gen_step
        max_path_len = self._max_path_len

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

            if spec_tree_manager is None:
                # CUDA graph warmup: accept only the first token per request
                num_accepted_tokens[num_contexts:batch_size] = 1
                accepted_tokens[num_contexts:batch_size, 0] = target_predict[:, 0].to(torch.int32)
                self._accepted_draft_indices_tensor[num_contexts:batch_size] = -1
                return accepted_tokens, num_accepted_tokens

            candidates = self._candidates_buf[:num_gens]
            candidates[:, 1:] = spec_metadata.draft_tokens.reshape(num_gens, N - 1).to(torch.int64)
            candidates[:, 0] = target_predict[:, 0]

            # Slots for gen rows: real py_seq_slot vs dummy; dummy -> slot_has_tree False.
            gen_slot_ids = spec_tree_manager._all_slot_ids_buf[
                num_contexts : num_contexts + num_gens
            ]
            tree_valid = spec_tree_manager.slot_has_tree[gen_slot_ids]

            _, accept_index, accept_token_num, accept_token = (
                self.tree_ops_converter.verify_dynamic_tree_greedy_out(
                    candidates,
                    spec_tree_manager.retrieve_index[:num_gens],
                    spec_tree_manager.retrieve_next_token[:num_gens],
                    spec_tree_manager.retrieve_next_sibling[:num_gens],
                    target_predict,
                    num_gens,
                    self._max_path_len,
                    tree_valid=tree_valid,
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

        return accepted_tokens, num_accepted_tokens

    @nvtx_range("eagle3_dyn.sample")
    def sample(
        self, logits: torch.Tensor, max_top_k: int, draft_model=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """TopK sampling with softmax for dynamic tree."""
        topk_indices, topk_values = _sample_softmax_topk(logits, max_top_k)
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
        batch_size: int,
        attn_metadata: AttentionMetadata = None,
    ):
        """Update draft tokens and scores, write contiguously to buffer."""
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

            self.prepare_tree_mask_and_position_offset(cur_draft_idx, attn_metadata, None)

            return new_draft_scores
        else:
            # Fused: score accumulation → topk → gather → parent selection
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

            write_history_start_offset = self.K + (cur_draft_idx - 1) * self.K * self.K
            write_history_end_offset = write_history_start_offset + self.K * self.K
            self.history_draft_tokens_buffer[
                :batch_size, write_history_start_offset:write_history_end_offset
            ] = new_draft_tokens
            self.history_score_buffer[
                :batch_size, write_history_start_offset:write_history_end_offset
            ] = new_draft_scores

            self._last_selected_parents = selected_parents
            self.prepare_tree_mask_and_position_offset(
                cur_draft_idx, attn_metadata, selected_parents
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
        return _resample_final_tokens(
            self.history_score_buffer[:batch_size, :],
            self.history_draft_tokens_buffer[:batch_size, :],
            self.max_total_draft_tokens,
        )

    def prepare_tree_mask_and_position_offset(
        self,
        cur_draft_idx: int,
        attn_metadata: AttentionMetadata,
        selected_parents: torch.Tensor = None,
    ):
        """Prepare the mask and position offsets for the next layer."""
        if attn_metadata.spec_decoding_packed_mask is None:
            return
        spec_tree_manager = self.spec_tree_manager
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

        # Hopper XQA needs packed 1D mask; Blackwell expects padded 3D.
        if self._needs_mask_repack:
            self._repack_mask_padded_to_packed(
                attn_metadata.spec_decoding_packed_mask, batch_size, num_tokens_current_layer
            )

    def prepare_for_generation(
        self,
        cur_draft_idx: int,
        attn_metadata: AttentionMetadata,
        batch_size: int,
        *,
        inputs=None,
        gather_ids=None,
        num_contexts: int = 0,
        num_gens: int = 0,
        num_accepted_tokens=None,
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
                        self._max_path_len
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
        step0_hs=None,
        hidden_states_to_save=None,
        selected_parents=None,
    ):
        """Manage hidden states for the growing context pattern."""
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

            # 3) Gather from write_buffer into accumulated_hs (positions K onwards).
            #    Positions 0:K retain step0_hs set in step 0 and are never modified.
            num_tokens_next = (cur_draft_idx + 1) * self.K
            read_idx = self._hs_read_map[:batch_size, self.K : num_tokens_next]
            gathered = torch.gather(
                self._hs_write_buffer[:batch_size],
                1,
                read_idx.unsqueeze(-1).expand(-1, -1, self._hs_dim),
            )
            self._accumulated_hs[:batch_size, self.K : num_tokens_next] = gathered
