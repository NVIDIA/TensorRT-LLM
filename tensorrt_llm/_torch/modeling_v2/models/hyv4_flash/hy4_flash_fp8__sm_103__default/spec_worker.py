# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFly2 drafter's worker: the block's inputs, its KV bookkeeping and its sampling walk.

The drafter keeps no arena: its block is appended to the target sparse layers' pools, so every step of
`DFlashWorker` that talks about the private arena is replaced. The native prologue ropes row `t` at
`sequence_length - width + t`, so a block at `[anchor, anchor + width)` runs with
`sequence_length = anchor + width`, which is also the length the engine expects after the step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.speculative.dflash import (
    DFlashSpecMetadata,
    DFlashWorker,
    dflash_noise_block_embedding,
)
from tensorrt_llm.logger import logger

from .draft import DflyBlockSelection
from .ops.spec.dfly_block_prepare import dfly_block_indices, dfly_block_prepare


class Hyv4FlashDFlashWorker(DFlashWorker):
    """The DFly2 drafter's worker at one rank of one deployment."""

    def __init__(self, spec_config, mapping, use_separate_draft_kv_cache: bool = False):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        self._dfly_inited = False

    def bind_target_lm_head(self, lm_head: nn.Module, logits_processor: nn.Module) -> None:
        """Keep the target logits gathered: the correction walk draws from the full vocabulary."""
        del lm_head, logits_processor
        self._vocab_parallel = None

    def _resolve_mask_token_id(self, draft_model) -> int:
        if self._resolved_mask_token_id is not None:
            return self._resolved_mask_token_id
        if getattr(self.spec_config, "mask_token_id", None) is not None:
            self._resolved_mask_token_id = self.spec_config.mask_token_id
        elif hasattr(draft_model, "mask_token_id"):
            self._resolved_mask_token_id = draft_model.mask_token_id
        elif hasattr(draft_model, "model") and hasattr(draft_model.model, "mask_token_id"):
            self._resolved_mask_token_id = draft_model.model.mask_token_id
        else:
            raise ValueError(
                "DFlash requires mask_token_id to be set. Please set it in DFlashDecodingConfig "
                "or ensure the draft model config has 'dflash_config.mask_token_id' or 'mask_token_id'."
            )
        return self._resolved_mask_token_id

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        """Allocate only the state `_forward_impl` snapshots every step; there is no arena.

        `_ctx_buf_inited` stays False, which keeps `DFlashSpecMetadata.prepare()`'s slot
        bookkeeping inert.
        """
        if self._dfly_inited:
            return
        del draft_kv_cache_manager
        self._validate_draft_attention_backend(draft_model)

        self._resolved_block_size = getattr(draft_model, "block_size", None) or (
            self.max_draft_len + 1
        )
        self._compute_block_size = self._draft_block_width(draft_model)
        if self._compute_block_size > self._resolved_block_size:
            slack = self._compute_block_size - self.max_draft_len
            raise ValueError(
                f"DFlash checkpoint was trained with block_size={self._resolved_block_size}."
                f"Lower max_draft_len to at most {self._resolved_block_size - slack}."
                f"Current max_draft_len={self.max_draft_len}."
            )
        self._resolve_mask_token_id(draft_model)

        cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        runtime_max = getattr(cache_manager, "max_seq_len", None)
        if runtime_max is None:
            runtime_max = getattr(attn_metadata, "max_seq_len", None)
        config = getattr(draft_model, "config", None)
        max_pos = getattr(config, "max_position_embeddings", None) if config else None
        candidates = [c for c in (runtime_max, max_pos) if c is not None]
        self._max_ctx = min(candidates) if candidates else 8192

        num_req_slots = spec_metadata.num_seq_slots
        self._dummy_slot = num_req_slots
        num_slots = num_req_slots + 1

        # Imported lazily to avoid an import cycle.
        from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

        self._graph_dummy_id_floor = CUDA_GRAPH_DUMMY_REQUEST_ID - self.max_draft_len

        self._ctx_len = torch.zeros(num_slots, dtype=torch.long, device="cuda")
        self._ctx_len_host = [0] * num_slots
        self._ctx_base_pages = torch.zeros(num_slots, dtype=torch.long, device="cuda")
        self._ctx_base_host = [0] * num_slots

        logger.info(
            f"DFly2: block width {self._compute_block_size}, mask token "
            f"{self._resolved_mask_token_id}, {num_slots} request slots, "
            f"served length {self._max_ctx}"
        )
        self._dfly_inited = True

    def _store_prefill_context(
        self,
        draft_model,
        spec_metadata: DFlashSpecMetadata,
        attn_metadata,
        position_ids: torch.Tensor,
        total_target_tokens: int,
    ):
        """Nothing: the prompt is already in the target's pool and its picks are published."""
        return None

    def _prepare_kv_for_draft_forward(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        """`prepare_1st_drafter_inputs` sets `kv_lens_cuda` to its final value, so the rewind is a no-op."""
        del num_accepted_tokens, num_contexts, batch_size
        self._kv_rewind_amount = None
        self._kv_rewind_pending = getattr(attn_metadata, "kv_lens_cuda", None) is not None

    def _write_block_kv_lens(
        self,
        attn_metadata,
        num_contexts: int,
        batch_size: int,
        anchors: torch.Tensor,
        width: int,
    ):
        """Point each generation row at the block's exclusive end, `anchors + width`.

        The sparse attention's per-row key count is `lengths`, not this; the prologue reads this.
        """
        new_lens = (anchors + width).to(torch.int32)
        kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
        if kv_lens is not None:
            kv_lens[num_contexts:batch_size] = new_lens
        # A warmup clone splits the runtime alias from `kv_lens_cuda`; write both when distinct.
        runtime = getattr(attn_metadata, "kv_lens_cuda_runtime", None)
        if runtime is not None and runtime is not kv_lens and runtime.shape[0] >= batch_size:
            runtime[num_contexts:batch_size] = new_lens
        for name in ("_seq_lens", "_seq_lens_cuda"):
            seq_lens = getattr(attn_metadata, name, None)
            if seq_lens is None or seq_lens.shape[0] < batch_size:
                continue
            seq_lens[num_contexts:batch_size] = width

    def prepare_1st_drafter_inputs(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        accepted_tokens: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        attn_metadata,
        spec_metadata: DFlashSpecMetadata,
        draft_model: nn.Module,
        total_target_tokens: int = 0,
    ):
        """The block: the anchor's bonus token then mask tokens, at their target positions and pool slots."""
        del input_ids, total_target_tokens
        d = draft_model.dims
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts
        width = self._compute_block_size
        device = hidden_states.device

        if num_gens == 0:
            # Called unconditionally; the drafter runs only with generation requests.
            return {
                "noise_embedding": hidden_states.new_empty(0, 0, d.hidden),
                "query_positions": torch.empty(0, 0, dtype=torch.long, device=device),
                "num_ctx_per_req": None,
                "ctx_k_cache": None,
                "ctx_v_cache": None,
                "ctx_cache_batch_idx": None,
                "ctx_kv_cache": None,
                "ctx_page_table": None,
                "block_placement": None,
                "first_prev_tokens": torch.empty(0, dtype=torch.long, device=device),
                "lilicorr_anchor": None,
            }

        mask_token_id = int(self._resolve_mask_token_id(draft_model))
        accepted = num_accepted_tokens[num_contexts:batch_size].to(torch.int64)
        num_drafts = (accepted - 1).clamp_min(0)
        bonus = (
            accepted_tokens[num_contexts:batch_size]
            .gather(1, num_drafts.unsqueeze(1))
            .squeeze(1)
            .long()
        )
        query_start_loc = torch.arange(
            num_gens + 1, dtype=torch.int64, device=device
        ) * width + int(attn_metadata.num_ctx_tokens)
        target_positions = position_ids.to(torch.int64).contiguous()
        if target_positions.dim() != 1:
            raise ValueError(
                "DFly2's block is built from the target batch's scalar positions; "
                f"got position_ids of shape {tuple(target_positions.shape)}"
            )

        manager = attn_metadata.kv_cache_manager
        page_size = int(manager.tokens_per_block)
        # One block per draft layer: the layers' pool rows differ, so only the slots depend on the
        # layer; positions, anchors and source rows are the first layer's.
        block = None
        layer_slots = []
        for layer in range(d.num_layers):
            scale, layer_offset = manager.get_primary_pool_page_index_params(
                draft_model._pool_row(attn_metadata, layer)
            )
            table = (
                attn_metadata.block_table[num_contexts:batch_size].to(torch.int64) * int(scale)
                + int(layer_offset)
            ).to(torch.int32)
            layer_block = dfly_block_prepare(
                bonus,
                num_drafts.to(torch.int32),
                query_start_loc,
                target_positions,
                table,
                int(self._max_ctx),
                mask_token_id=mask_token_id,
                block_width=width,
                block_size=page_size,
            )
            if block is None:
                block = layer_block
            layer_slots.append(layer_block.slots.view(num_gens, width))

        self._write_block_kv_lens(attn_metadata, num_contexts, batch_size, block.anchors, width)
        # Rebuild the indexer map, write slots and scheduler buffers from the lengths just written.
        attn_metadata.update_for_spec_dec()

        picks = manager.draft_picks(d.target_kv_layers, d.index_topk, device=device)
        rows = num_gens * width
        packed = torch.empty((d.num_layers, rows, d.index_width), dtype=torch.int32, device=device)
        lengths = torch.empty((d.num_layers, rows), dtype=torch.int32, device=device)
        for layer in range(d.num_layers):
            dfly_block_indices(
                picks.history[layer : layer + 1],
                block.source_rows,
                layer_slots[layer],
                block.active,
                packed[layer : layer + 1],
                lengths[layer : layer + 1],
            )

        embed_weight = draft_model.embed_tokens_weight
        trained_mask_embedding = (
            getattr(draft_model, "mask_token_embedding", None)
            if mask_token_id == getattr(draft_model, "mask_token_id", None)
            else None
        )
        noise_embedding = dflash_noise_block_embedding(
            lambda ids: F.embedding(ids, embed_weight),
            bonus,
            mask_token_id,
            width,
            trained_mask_embedding,
        )
        target_hidden = hidden_states.index_select(
            0, block.source_rows.clamp(max=hidden_states.shape[0] - 1)
        )
        if target_hidden.dtype != noise_embedding.dtype:
            target_hidden = target_hidden.to(noise_embedding.dtype)
        noise_embedding[:, 0, :] = draft_model.project_target_hidden(
            noise_embedding[:, 0, :], target_hidden
        )

        return {
            "noise_embedding": noise_embedding,
            "query_positions": block.positions.view(num_gens, width),
            "num_ctx_per_req": None,
            "ctx_k_cache": None,
            "ctx_v_cache": None,
            "ctx_cache_batch_idx": None,
            "ctx_kv_cache": None,
            "ctx_page_table": None,
            "block_placement": DflyBlockSelection(
                packed=packed,
                lengths=lengths,
                slots=layer_slots[0],
                positions=block.positions,
                active=block.active,
                attn_metadata=attn_metadata,
            ),
            "first_prev_tokens": bonus,
            "lilicorr_anchor": None,
        }

    def _sample_block_draft_tokens(
        self,
        draft_model: nn.Module,
        gen_logits: torch.Tensor,
        gen_hidden_states: torch.Tensor,
        inputs: dict,
        spec_metadata: DFlashSpecMetadata,
        num_contexts: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, int]:
        """Draw the `K` drafts in order: column `s` is corrected with column `s - 1`'s token first.

        `gen_logits` only supplies shapes; each step's logits come from the corrected row.
        """
        num_gens, num_drafts = gen_logits.shape[:2]
        vocab_size = gen_logits.shape[-1]
        del gen_logits

        attn_metadata = inputs["block_placement"].attn_metadata
        block_hidden = gen_hidden_states.reshape(num_gens, num_drafts, -1)
        previous = inputs["first_prev_tokens"]

        draw_with_probs = spec_metadata.wants_advanced_draft_sampling
        if draw_with_probs:
            sampling_params = self._draft_sampling_params(spec_metadata, num_contexts, batch_size)
            slots = (
                None
                if spec_metadata.draft_probs is None
                else spec_metadata.batch_slot_ids[num_contexts:batch_size]
            )
            seed = spec_metadata.request_seeds[num_contexts:batch_size]
            offsets = (
                self._rng_state_per_request(
                    spec_metadata,
                    num_contexts,
                    batch_size,
                    repeat=num_drafts,
                    slot=spec_metadata.rng_draft_slot(0),
                )[1]
                .view(num_gens, num_drafts)
                .t()
                .contiguous()
            )

        drawn = []
        for step in range(num_drafts):
            corrected, _confidence = draft_model.hidden_correction(block_hidden[:, step], previous)
            logits = draft_model.block_logits(corrected, attn_metadata)
            if draw_with_probs:
                logits = self._zero_padding_rows(logits, spec_metadata, num_contexts, batch_size, 1)
                tokens, probs = self._sample_from_logits_with_probs(
                    spec_metadata.advanced_sampling_mode,
                    logits,
                    *sampling_params,
                    seed=seed,
                    offset=offsets[step],
                )
                if slots is not None:
                    spec_metadata.draft_probs[slots, step, :vocab_size] = probs
            else:
                tokens = self.greedy_sample_draft_with_tp_gather(logits, spec_metadata)
            drawn.append(tokens.to(torch.int32))
            previous = tokens

        if draw_with_probs and spec_metadata.draft_probs is not None:
            spec_metadata.draft_probs_last_dim = vocab_size
        return torch.stack(drawn, dim=1).to(torch.int32), vocab_size


__all__ = ["Hyv4FlashDFlashWorker"]
