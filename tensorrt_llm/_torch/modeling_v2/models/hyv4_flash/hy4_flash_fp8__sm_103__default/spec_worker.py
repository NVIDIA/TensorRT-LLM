# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFly2 drafter's worker: the block's inputs, its KV bookkeeping and its sampling walk.

The engine's `DFlashWorker` drives a drafter that keeps a context of its own:
it captures the target's prefill hidden states into a private arena, appends
each step's block to that arena, and reads it back. This drafter keeps nothing
-- it appends its block to the *target* sparse layer's latent pool, at the rows
its own block's positions name, and reads that layer's pick list back. The
worker therefore replaces every step of the engine's preparation that talks
about the private arena, and keeps the parts that are about the target:

* `_lazy_init_ctx_buffers` builds only the state `_forward_impl` reads
  unconditionally (`_ctx_len` and its snapshots), never the arena or its
  buffers, and resolves the block width and the mask token from the drafter.
  `_ctx_buf_inited` stays False, which keeps `DFlashSpecMetadata.prepare()`'s
  slot bookkeeping inert: this drafter has no request slots.
* `prepare_1st_drafter_inputs` builds the block itself -- `DflyBlockPrepare`
  for the six rows' positions and pool slots, the family manager's
  `draft_picks` state plus `DflyBlockIndices` for each draft layer's selection,
  and the slot-0 fusion of the target's own hidden state.
* `_sample_block_draft_tokens` walks the block: DFly2's draft positions
  condition on the previous draft through the hidden correction, so the five
  drafts of a step are drawn one after another, each from the corrected hidden
  of its own block column.
* `_prepare_kv_for_draft_forward` leaves `kv_lens_cuda` alone and only records
  that a rewind is pending; this worker's own write sets it to the block's last
  position, which is exactly the value the engine expects at the end of a step
  (the target's own sequence length advanced by the accepted tokens), so the
  deferred rewind resolves to nothing.

The two halves of the KV bookkeeping are one convention: the native prologue
ropes a generation row as `sequence_length[request] - predicted_tokens_per_seq
+ t`, so a block whose rows sit at `[anchor, anchor + width)` needs
`sequence_length = anchor + width` while it runs -- and that value, with
`anchor = <the target's sequence length> - width + num_accepted_tokens`, is the
value the engine's own step would have left for the next one.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm._torch.speculative.dflash import (
    DFlashSpecMetadata,
    DFlashWorker,
    dflash_draft_slot_ids,
    dflash_noise_block_embedding,
)
from tensorrt_llm.logger import logger

from .draft import DflyBlockSelection
from .ops.spec.dfly_block_prepare import dfly_block_indices, dfly_block_prepare


class Hyv4FlashDFlashWorker(DFlashWorker):
    """The DFly2 drafter's worker at one rank of one deployment.

    Everything step-shaped comes from the engine; what this class owns is the
    block: where its rows live in the target's pool and what they attend, how
    its five drafts are sampled, and the sequence lengths that follow from it.
    """

    def __init__(self, spec_config, mapping, use_separate_draft_kv_cache: bool = False):
        super().__init__(spec_config, mapping, use_separate_draft_kv_cache)
        # The base's `_ctx_buf_inited` guards its private-arena state; this
        # worker's state is the four snapshotted tensors, allocated once.
        self._dfly_inited = False

    # -- what the engine binds onto the worker ---------------------------------

    def bind_target_lm_head(self, lm_head: nn.Module, logits_processor: nn.Module) -> None:
        """Keep the target logits gathered: this drafter's walk reads full vocab.

        The engine's DFlash binds a vocabulary-shard selection over the target
        head when the processor is the plain one, and `target_logits` then hands
        back this rank's shard. DFly2's drafts are drawn from the drafter's own
        full-vocabulary head (`Hyv4FlashDraft.block_logits`, over the corrected
        hidden of each block column), and the acceptance reads the same rows, so
        a second, shard-shaped source of truth is declined here: the worker
        verifies through the gathered logits (`verifies_on_target_shard`), which
        is also why `_vocab_parallel_draft_tokens` returns None.
        """
        del lm_head, logits_processor
        self._vocab_parallel = None

    # -- geometry the engine resolves off the drafter --------------------------

    def _resolve_mask_token_id(self, draft_model) -> int:
        """The drafter's mask token, from the config, the drafter, or its model."""
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
        """Build the state `_forward_impl` snapshots; the drafter keeps no arena.

        The engine's version sizes a private context pool from the served
        length and builds the drafter's fused KV buffers. This drafter writes
        into the target's pool instead, so none of that exists -- but
        `_forward_impl` reads and snapshots four tensors on every step
        regardless of the drafter, so they are allocated here, once, at the
        slot count the engine's spec metadata reports. The manager arguments
        are unused: this worker addresses the target's pool through the
        metadata, never a draft manager of its own.
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

        # The block's positions are bounded by the served length, taken from
        # the KV cache manager (as `DFlashWorker` does), with the model's own
        # advertised length as the fallback.
        cache_manager = getattr(attn_metadata, "kv_cache_manager", None)
        runtime_max = getattr(cache_manager, "max_seq_len", None)
        if runtime_max is None:
            runtime_max = getattr(attn_metadata, "max_seq_len", None)
        config = getattr(draft_model, "config", None)
        max_pos = getattr(config, "max_position_embeddings", None) if config else None
        candidates = [c for c in (runtime_max, max_pos) if c is not None]
        self._max_ctx = min(candidates) if candidates else 8192

        # The engine's `DFlashSpecMetadata.prepare()` guards its slot
        # bookkeeping on `_ctx_buf_inited`, which stays False: a slot is
        # private-arena state.
        num_req_slots = spec_metadata.num_seq_slots
        self._dummy_slot = num_req_slots
        num_slots = num_req_slots + 1

        # CUDA-graph padding uses ids counting down from
        # CUDA_GRAPH_DUMMY_REQUEST_ID; a floor of one draft block separates
        # them from real ids. Imported lazily to avoid an import cycle.
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
        """Nothing: this drafter captures no prefill context.

        The engine's drafter projects the target's prefill hidden states into
        its own arena so a request's first blocks can attend the prompt. Here
        the prompt is already in the target sparse layer's pool -- the same
        pool the block is written into -- and the target's indexer publishes
        each context request's last row in the picks state, so there is nothing
        to capture; skipping it also skips the engine's request-slot
        assignment, which is why `_ctx_buf_inited` stays False.
        """
        return None

    # -- KV bookkeeping -------------------------------------------------------

    def _prepare_kv_for_draft_forward(
        self,
        attn_metadata,
        num_accepted_tokens: torch.Tensor,
        num_contexts: int,
        batch_size: int,
    ):
        """Record that this step's `kv_lens_cuda` is this worker's to set.

        The engine's drafter raises the length by one before the draft forward
        and rewinds it after, because its block sits one position past the
        target's own. This block's rows sit at `anchor`, so
        `prepare_1st_drafter_inputs` writes the length the block's own forward
        needs -- which is also the value the engine expects at the end of the
        step -- and the deferred rewind, whose only purpose is to keep a failed
        forward from leaving a raised length, resolves to nothing. The accepted
        counts are not needed here: the block is built from the target batch's
        own positions.
        """
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
        """Point each generation row at the block's last position.

        `anchors + width` is the block's exclusive end: the native prologue
        ropes row `t` of the block at `sequence_length - width + t`, and the
        block's rows all sit at or before it. It is also exactly where the
        engine's own step leaves the length -- the target's own
        `sequence_length` advanced by the tokens this step accepted -- so
        nothing has to be compensated after the draft forward.

        It is *not* the count the sparse generation attention works to: that is
        each row's own live prefix, the `lengths` this module writes beside
        `packed`, which the block's forward hands the kernel as the request's
        bound (`draft.Hyv4FlashDraft._block_attention`). The two part once a
        request's picks fill its selection, and the buffer here keeps the
        pool's end because the prologue is what reads it.
        """
        new_lens = (anchors + width).to(torch.int32)
        kv_lens = getattr(attn_metadata, "kv_lens_cuda", None)
        if kv_lens is not None:
            kv_lens[num_contexts:batch_size] = new_lens
        # The metadata's runtime aliases point at the same buffers the kernels
        # read; a warmup clone splits them, so both are written when distinct.
        # The clone carries the whole batch, context rows included.
        runtime = getattr(attn_metadata, "kv_lens_cuda_runtime", None)
        if runtime is not None and runtime is not kv_lens and runtime.shape[0] >= batch_size:
            runtime[num_contexts:batch_size] = new_lens
        # The step's sequence lengths: the engine's drafter sets the window's
        # token count the same way, and the blocks built from it follow.
        for name in ("_seq_lens", "_seq_lens_cuda"):
            seq_lens = getattr(attn_metadata, name, None)
            if seq_lens is None or seq_lens.shape[0] < batch_size:
                continue
            seq_lens[num_contexts:batch_size] = width

    # -- the block's inputs ---------------------------------------------------

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
        """Everything the drafter's block forward reads, this step.

        The block is `block_width` rows per generation request: the accepted
        anchor's bonus token followed by mask tokens, all of them written into
        the target sparse layer's pool at the positions they occupy in the
        target's own sequence. `DflyBlockPrepare` gives those rows and their
        slots, `DflyBlockIndices` gives each draft layer's selection over that
        layer's picks, and the anchor's column is fused with the target's own
        hidden state for that row.
        """
        del input_ids, total_target_tokens
        d = draft_model.dims
        num_contexts = attn_metadata.num_contexts
        batch_size = attn_metadata.num_seqs
        num_gens = batch_size - num_contexts
        width = self._compute_block_size
        device = hidden_states.device

        if num_gens == 0:
            # A context-only step: the engine calls this preparation
            # unconditionally and runs the drafter only when there are
            # generation requests.
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
        # The anchor token: the last token the target accepted for this
        # request, which the block's first column carries as its input.
        bonus = (
            accepted_tokens[num_contexts:batch_size]
            .gather(1, num_drafts.unsqueeze(1))
            .squeeze(1)
            .long()
        )
        # The anchor is the first row after the accepted ones, so a block's
        # rows start at `query_start_loc + num_accepted` -- one row past the
        # target's last accepted row, whose hidden predicts the bonus.
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
        # One block per draft layer, because the sparse layers' pool rows
        # differ: `scale`/`layer_offset` map a request's page to that layer's
        # rows, and only the slots depend on the table. The rows themselves --
        # positions, anchors and the block's source row -- are shared, so the
        # first layer's block supplies them.
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
        # The indexer's per-token map, the write slots and the scheduler
        # buffers are rebuilt from the lengths just written.
        attn_metadata.update_for_spec_dec()

        picks = manager.draft_picks(d.target_kv_layers, d.index_topk, device=device)
        rows = num_gens * width
        packed = torch.empty((d.num_layers, rows, d.index_width), dtype=torch.int32, device=device)
        lengths = torch.empty((d.num_layers, rows), dtype=torch.int32, device=device)
        for layer in range(d.num_layers):
            # One layer at a time: the entry reads a per-layer pick pool and
            # writes a per-layer slice, and the slices keep the state's strides.
            dfly_block_indices(
                picks.history[layer : layer + 1],
                block.source_rows,
                layer_slots[layer],
                block.active,
                packed[layer : layer + 1],
                lengths[layer : layer + 1],
            )

        # The block's rows: the bonus token's embedding with the target's own
        # hidden state fused into column 0, mask tokens elsewhere.
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
                # Only the slots depend on the layer's table; the rest of the
                # block is the same rows for all five.
                slots=layer_slots[0],
                positions=block.positions,
                active=block.active,
                attn_metadata=attn_metadata,
            ),
            # The anchor token per generation request: the walk conditions its
            # first position on it.
            "first_prev_tokens": bonus,
            "lilicorr_anchor": None,
        }

    # -- the block's drafts ---------------------------------------------------

    def _draft_slot_ids(
        self, draft_model, num_gens: int, block_size: int, num_draft_tokens: int
    ) -> torch.Tensor:
        """The block's columns `1 .. K`: the hidden that predicts each column's own token.

        DFly2's correction conditions column `s`'s hidden on the token of
        column `s - 1`, so the corrected hidden of column `s` is the row whose
        logits are column `s`'s own draft (see `_sample_block_draft_tokens`).
        Column 0 is the anchor, whose token the target has already produced, so
        it is never drafted from -- the engine's plain-DFlash convention, which
        the base class already asks for.
        """
        del draft_model
        return dflash_draft_slot_ids(num_gens, block_size, num_draft_tokens, False, device="cuda")

    def _vocab_parallel_draft_tokens(self, draft_model) -> Optional[object]:
        """None: the shard's fast path reads hidden the correction has not walked.

        The engine's greedy draft path takes block logits straight from this
        rank's vocabulary shard. DFly2's drafts condition on each other, so
        that path would answer from uncorrected hidden states; declining it
        keeps the correction walk the only route to a draft token.
        """
        del draft_model
        return None

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
        """Walk the block: `K` drafts, each conditioned on the ones before it.

        The engine's DFlash draws every position independently from its block
        logits. DFly2 corrects column `s`'s hidden with the embedding of the
        token at column `s - 1` before that column's logits are taken, so
        column `s`'s draft depends on column `s - 1`'s, which is only known
        once it is drawn. The walk is therefore one step per column, in order,
        from the anchor token: step `s` corrects the gathered row of column
        `s + 1` with the token at column `s` and draws column `s + 1`'s token.
        Each step's logits stand in for the precomputed block logits of that
        column: `gen_logits` is dropped, not refined.
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
