# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP8 CTXLEN_FUSE: this target's copy of the in-tree DFlash first drafter inputs
(``DFlashWorker.prepare_1st_drafter_inputs``) with the stored-context resume update in one launch.

:func:`prepare_1st_drafter_inputs_ctxlen_fused` is the in-tree method line for line except the manager-bound pool's
resume update (``ctx_len[slots] = where(ctx_stored[slots], position_ids[first_input].clamp(max_ctx), ctx_len[slots])``,
nine eager kernels), which :func:`resume_ctx_len` runs as one CUDA C++ launch (``kernels.m3_ctxlen_resume``; integer
only, bitwise). Operands outside that launch's contract take the in-tree eager lines. The unit test
``test_m31_tp8_ctxlen_fuse`` checks that this copy still matches the in-tree method apart from that block.

Default ON for this target; kill switch ``TRTLLM_M31_DISABLE_TP8_CTXLEN_FUSE=1`` (read at import) restores the
in-tree method.
"""

from __future__ import annotations

import torch
from torch import nn

from tensorrt_llm._torch.attention.backends import AttentionMetadata
from tensorrt_llm._torch.speculative.dflash import (
    DFlashSpecMetadata,
    dflash_allocated_ctx_limit,
    dflash_noise_block_embedding,
    dflash_rebased_ctx_view,
    last_accepted_hidden,
)
from tensorrt_llm.logger import logger

from .kernels import m3_ctxlen_resume
from .m31_switches import m31_disabled

TP8_CTXLEN_FUSE = not m31_disabled("TP8_CTXLEN_FUSE")

_LOGGED: set = set()


def _log_once(key: str, message: str) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(message)


# An in-tree DFlash worker helper the copy below calls; a worker without it runs another version of the method.
_WORKER_HELPERS = ("_static_arange",)


def prepare(device: int, worker=None) -> bool:
    """Compile the resume kernels for ``device`` (outside capture). Whether the item is on for this rank: off when
    ``worker`` lacks the in-tree helpers :func:`prepare_1st_drafter_inputs_ctxlen_fused` copies."""
    if not TP8_CTXLEN_FUSE:
        return False
    missing = [name for name in _WORKER_HELPERS if worker is not None and not hasattr(worker, name)]
    if missing:
        _log_once(
            "worker",
            f"[M3 dispatch sm_107 tp8] TP8 CTXLEN_FUSE off: the DFlash worker has no {', '.join(missing)}",
        )
        return False
    reason = m3_ctxlen_resume.prepare(device)
    if reason is not None:
        _log_once("prep", f"[M3 dispatch sm_107 tp8] TP8 CTXLEN_FUSE off: {reason}")
        return False
    return True


def resume_ctx_len(
    worker, position_ids, num_ctx_tokens: int, tokens_per_req: int, slots, num_gens: int
) -> None:
    """The stored-context resume update: the one-launch kernel when it takes the operands, else the in-tree lines."""
    reason = m3_ctxlen_resume.outside_contract(
        position_ids, slots, worker._ctx_stored, worker._ctx_len
    )
    if reason is None:
        m3_ctxlen_resume.ctxlen_resume_(
            position_ids,
            slots,
            worker._ctx_stored,
            worker._ctx_len,
            num_ctx_tokens,
            tokens_per_req,
            num_gens,
            worker._max_ctx,
        )
        if torch.cuda.is_current_stream_capturing():
            rank = getattr(getattr(worker, "mapping", None), "tp_rank", None)
            _log_once(
                "taken",
                f"[M3 dispatch sm_107 tp8] TP8 CTXLEN_FUSE taken: drafter ctx_len resume in one launch "
                f"(captured, {num_gens} requests, rank {rank})",
            )
        return
    _log_once(
        f"not:{reason}",
        f"[M3 dispatch sm_107 tp8] TP8 CTXLEN_FUSE not taken ({reason}); the eager resume runs",
    )
    first_input = num_ctx_tokens + tokens_per_req * worker._static_arange(0, num_gens)
    worker._ctx_len[slots] = torch.where(
        worker._ctx_stored[slots],
        position_ids[first_input].long().clamp(max=worker._max_ctx),
        worker._ctx_len[slots],
    )


def prepare_1st_drafter_inputs_ctxlen_fused(
    self,
    input_ids: torch.LongTensor,
    position_ids: torch.LongTensor,
    hidden_states: torch.Tensor,
    accepted_tokens: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    attn_metadata: AttentionMetadata,
    spec_metadata: DFlashSpecMetadata,
    draft_model: nn.Module,
    total_target_tokens: int = 0,
):
    """Prepare inputs for DFlash's draft forward.

    For gen requests, builds:
    - noise_embedding: token embeddings for [accepted + mask] tokens
    - query_positions: position IDs for the query tokens
    - num_ctx_per_req: per-request context length in the pool
    - ctx_k_cache / ctx_v_cache / ctx_cache_batch_idx: slot-indexed
      views of the persistent per-layer K/V pool.
    """
    num_contexts = attn_metadata.num_contexts
    batch_size = attn_metadata.num_seqs
    num_gens = batch_size - num_contexts

    # Resolve mask_token_id and block_size once, cache for subsequent calls
    if self._resolved_mask_token_id is None:
        if (
            hasattr(self.spec_config, "mask_token_id")
            and self.spec_config.mask_token_id is not None
        ):
            self._resolved_mask_token_id = self.spec_config.mask_token_id
        elif hasattr(draft_model, "mask_token_id"):
            self._resolved_mask_token_id = draft_model.mask_token_id
        elif hasattr(draft_model.model, "mask_token_id"):
            self._resolved_mask_token_id = draft_model.model.mask_token_id
        else:
            raise ValueError(
                "DFlash requires mask_token_id to be set. Please set it in DFlashDecodingConfig "
                "or ensure the draft model config has 'dflash_config.mask_token_id' or 'mask_token_id'."
            )
    mask_token_id = self._resolved_mask_token_id

    # Get the embed_tokens layer from the draft model
    embed_tokens = draft_model.draft_model_full.model.embed_tokens
    hidden_dim = (
        spec_metadata.hidden_size if spec_metadata.hidden_size > 0 else hidden_states.shape[-1]
    )
    lilicorr_anchor = None

    if num_gens > 0:
        gen_num_accepted = num_accepted_tokens[num_contexts : num_contexts + num_gens]
        gen_accepted_tokens = accepted_tokens[num_contexts : num_contexts + num_gens, :]

        total_tokens_per_req = self._draft_tokens_per_req  # K+1
        K = spec_metadata.runtime_draft_len

        # Get captured multi-layer hidden states from spec_metadata
        captured_hs = spec_metadata.get_hidden_states(total_target_tokens)
        has_target_features = (
            captured_hs is not None
            and hasattr(draft_model, "fc")
            and hasattr(draft_model, "hidden_norm")
        )

        # Use cached block_size (resolved once on first call)
        block_size = self._compute_block_size
        query_tokens_per_req = block_size

        # Get slots for gen requests from pre-computed mapping
        slots = self._batch_to_slot[num_contexts : num_contexts + num_gens]
        gen_rows_out = self._static_arange(num_contexts, num_contexts + num_gens)

        K_plus_1 = K + 1

        if self._ctx_block_tables is not None:
            # Stored context sits at target positions, which also resumes a parked one (one launch).
            resume_ctx_len(
                self,
                position_ids,
                attn_metadata.num_ctx_tokens,
                total_tokens_per_req,
                slots,
                num_gens,
            )
        # A target's one-launch step inputs: the same bonus, positions and noise block as the stock lines below.
        fused_inputs = (
            self._fused_drafter_inputs(
                embed_tokens,
                gen_accepted_tokens,
                gen_num_accepted,
                slots,
                self._ctx_len,
                self._max_ctx,
                mask_token_id,
                getattr(draft_model, "mask_token_embedding", None)
                if mask_token_id == getattr(draft_model, "mask_token_id", None)
                else None,
                query_tokens_per_req,
                K_plus_1,
            )
            if self._fused_drafter_inputs is not None
            else None
        )
        if fused_inputs is not None:
            bonus, query_position_ids, ctx_position_ids, noise_embed_2d = fused_inputs
            offsets_kp1 = self._static_arange(0, K_plus_1)
        else:
            bonus_idx = (gen_num_accepted - 1).clamp_min(0).long().unsqueeze(1)
            bonus = gen_accepted_tokens.gather(1, bonus_idx).squeeze(1).long()

            ctx_len_gen = self._ctx_len[slots]
            j_block = self._static_arange(0, query_tokens_per_req)
            offsets_kp1 = self._static_arange(0, K_plus_1)

            # _ctx_len is clamped to _max_ctx only AFTER this step's accepted
            # tokens are folded in (see the update below), so the running length
            # used here has to be clamped on its own -- otherwise a request that
            # already sits at the ceiling indexes num_accepted positions past
            # any position the sequence can legitimately reach.
            ctx_len_now = (ctx_len_gen + gen_num_accepted.long()).clamp_(max=self._max_ctx)
            query_position_ids = ctx_len_now.unsqueeze(1) + j_block.unsqueeze(0)
            ctx_position_ids = ctx_len_gen.unsqueeze(1) + offsets_kp1.unsqueeze(0)

            # The drafter's own trained mask row, if it kept one for the mask id in use.
            trained_mask_embedding = (
                getattr(draft_model, "mask_token_embedding", None)
                if mask_token_id == getattr(draft_model, "mask_token_id", None)
                else None
            )
            noise_embed_2d = dflash_noise_block_embedding(
                embed_tokens, bonus, mask_token_id, query_tokens_per_req, trained_mask_embedding
            )

        # Accumulate new accepted features into context buffers
        fused_num_ctx = None
        if has_target_features:
            gen_start = attn_metadata.num_ctx_tokens
            # Target now processes exactly K+1 tokens per gen req, so the
            # captured slice is already the full set we need to project.
            gen_hs = captured_hs[gen_start : gen_start + num_gens * total_tokens_per_req]
            gen_hs_to_project = gen_hs.reshape(-1, gen_hs.shape[-1])
            projected_to_store = draft_model.project_target_hidden(gen_hs_to_project)
            if getattr(draft_model, "lilicorr", None) is not None:
                # The last accepted output is predicted by the corresponding input
                # row; the newly committed token itself has no target row yet.
                projected_rows = projected_to_store.reshape(num_gens, K_plus_1, -1)
                lilicorr_anchor = last_accepted_hidden(projected_rows, gen_num_accepted)
            # enable_int32_accept_counts: the fused store / advance take the int32 counts as is; the stock lines
            # below convert on first use.
            int32_counts = self._int32_accept_counts is not None
            fused_counts = gen_num_accepted if int32_counts else gen_num_accepted.long()
            gen_num_accepted_long = None if int32_counts else fused_counts
            # Fused context store: a target's fused store derives each row's column from its position itself (the stock
            # clamp below), so the column glue runs only on the stock path.
            fused_stored = False
            if self._fused_ctx_store is not None and self._ctx_paged:
                bound = self._ctx_block_tables is not None
                fused_stored = self._fused_ctx_store(
                    draft_model,
                    projected_to_store.to(self._ctx_kv_buf[0].dtype),
                    ctx_position_ids.long().reshape(-1),
                    fused_counts,
                    gen_rows_out if bound else slots,
                    None,
                    self._ctx_block_tables if bound else self._ctx_page_table,
                    self._ctx_kv_buf,
                    K + 1,
                    capacity_counts=self._ctx_block_counts if bound else None,
                    max_col=None if bound else self._max_ctx - 1,
                )
            int32_stored = int32_counts and fused_stored
            if not fused_stored:
                if gen_num_accepted_long is None:
                    gen_num_accepted_long = gen_num_accepted.long()
                col_idx = self._ctx_len[slots].unsqueeze(1) + offsets_kp1.unsqueeze(0)
                write_mask = offsets_kp1.unsqueeze(0) < gen_num_accepted_long.unsqueeze(1)
                if self._ctx_block_tables is not None:
                    # Bound by what the manager handed THIS request: the masked
                    # entries below are still written (fixed-size, for graph
                    # safety) and a placeholder page resolves to another's.
                    gen_capacity = self._ctx_block_counts[gen_rows_out] * self._ctx_page_size
                    col_idx = torch.minimum(col_idx, (gen_capacity - 1).unsqueeze(1)).clamp_(min=0)
                else:
                    col_idx = col_idx.clamp(max=self._max_ctx - 1)

                # Fixed-size writes for CUDA graph compatibility:
                # Write ALL entries but zero out invalid ones. Invalid
                # writes land at clamped column indices beyond valid
                # _ctx_len range, so they're harmless.
                slot_flat = slots.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                col_flat = col_idx.reshape(-1)
                proj_flat = projected_to_store  # already [num_gens*(K+1), proj_dim]
                pos_flat = ctx_position_ids.long().reshape(-1)
                mask_1d = write_mask.reshape(-1)

                # Fast path: store the pre-projected/pre-RoPE'd K/V.
                # dflash_forward reads these directly via cache_batch_idx.
                cache_dtype = (
                    self._ctx_kv_buf[0].dtype if self._ctx_paged else self._ctx_k_buf.dtype
                )
                # A target's fused store does the K / V epilogue, the accept mask and the paged write in one launch.
                fused_stored = (
                    self._fused_ctx_store is not None
                    and self._ctx_paged
                    and self._fused_ctx_store(
                        draft_model,
                        proj_flat.to(cache_dtype),
                        pos_flat,
                        gen_num_accepted_long,
                        gen_rows_out if self._ctx_block_tables is not None else slots,
                        col_flat,
                        self._ctx_block_tables
                        if self._ctx_block_tables is not None
                        else self._ctx_page_table,
                        self._ctx_kv_buf,
                        K + 1,
                    )
                )
                if not fused_stored:
                    k_new, v_new = draft_model.precompute_context_kv(
                        proj_flat.to(cache_dtype), pos_flat
                    )
                    mask_bc = mask_1d.view(-1, 1, 1, 1).to(k_new.dtype)
                    k_new.mul_(mask_bc)
                    if v_new is not None:
                        v_new.mul_(mask_bc)
                    slot_long = slot_flat.long()
                    col_long = col_flat.long()
                    if self._ctx_paged:
                        if self._ctx_block_tables is not None:
                            # Batch positions of the gen requests, matching the
                            # per-request block table's row order.
                            rows_long = gen_rows_out.unsqueeze(1).expand(-1, K + 1).reshape(-1)
                        else:
                            rows_long = slot_long
                        self._store_context_kv_paged(k_new, v_new, rows_long, col_long)
                    else:  # VANILLA DFlash backend (FlashAttention)
                        self._ctx_k_buf[slot_long, :, col_long] = k_new
                        if v_new is not None:
                            self._ctx_v_buf[slot_long, :, col_long] = v_new

            # A target's one-launch advance + read view: the stock lines' context lengths and read rows, or None.
            if self._fused_ctx_advance is not None and self._ctx_block_tables is not None:
                fused_num_ctx = self._fused_ctx_advance(
                    slots,
                    gen_rows_out,
                    fused_counts,
                    self._ctx_len,
                    self._max_ctx,
                    self._dummy_slot,
                    self._ctx_block_counts,
                    self._ctx_base_pages,
                    self._ctx_block_tables,
                    self._ctx_block_tables_read,
                    self._ctx_page_size,
                    block_size,
                )
            if fused_num_ctx is None:
                if gen_num_accepted_long is None:
                    gen_num_accepted_long = gen_num_accepted.long()
                self._advance_ctx_len(slots, gen_num_accepted_long)
            elif int32_stored:
                logger.info_once(
                    f"{self._int32_accept_counts} the int32 accept counts went to the fused context store and "
                    "advance (no int64 copy)",
                    key="dflash_int32_accept_counts",
                )

        if fused_num_ctx is not None:
            num_ctx_per_req_t = fused_num_ctx
            ctx_page_table = self._ctx_block_tables_read
        else:
            num_ctx_per_req_t = self._ctx_len[slots]
            ctx_page_table = self._ctx_page_table
            if self._ctx_block_tables is not None:
                # The write above clamps columns to the request's allocation, so
                # a context that outruns it has its tail written over the last
                # valid slot. Truncate what is advertised to match, or the read
                # attends that clobbered value -- or another request's page.
                allocated = dflash_allocated_ctx_limit(
                    self._ctx_block_counts, self._ctx_page_size, block_size
                )
                num_ctx_per_req_t = torch.minimum(num_ctx_per_req_t, allocated[gen_rows_out])
                # Pages below a slot's base hold no drafter K/V (a prefix the
                # target reused): the read numbers from the base instead, see
                # _ctx_base_for_store and dflash_rebased_ctx_view. Writes above
                # keep the absolute table.
                base_pages = self._ctx_base_pages[slots]
                self._ctx_block_tables_read[gen_rows_out], num_ctx_per_req_t = (
                    dflash_rebased_ctx_view(
                        self._ctx_block_tables[gen_rows_out],
                        base_pages,
                        num_ctx_per_req_t,
                        self._ctx_page_size,
                    )
                )
                ctx_page_table = self._ctx_block_tables_read
        noise_embedding = noise_embed_2d
        query_positions = query_position_ids.long()

        # Update seq_lens for gen requests to K+1 (the number of tokens
        # the target forward actually processed).
        attn_metadata._seq_lens_cuda[num_contexts : num_contexts + num_gens] = total_tokens_per_req
        attn_metadata._seq_lens[num_contexts : num_contexts + num_gens] = total_tokens_per_req
    else:
        noise_embedding = hidden_states.new_empty(0, 0, hidden_dim)
        query_positions = torch.empty(0, 0, dtype=torch.long, device="cuda")
        num_ctx_per_req_t = torch.empty(0, dtype=torch.long, device="cuda")
        slots = torch.empty(0, dtype=torch.long, device="cuda")
        gen_rows_out = slots
        bonus = torch.empty(0, dtype=torch.long, device="cuda")
        ctx_page_table = (
            self._ctx_block_tables_read
            if self._ctx_block_tables is not None
            else self._ctx_page_table
        )

    return {
        "lilicorr_anchor": lilicorr_anchor,
        "noise_embedding": noise_embedding,
        "query_positions": query_positions,
        "num_ctx_per_req": num_ctx_per_req_t,
        "ctx_k_cache": self._ctx_k_buf,
        "ctx_v_cache": self._ctx_v_buf,
        # Slots index the private arena; the manager's block table is keyed
        # by batch position, so the drafter reads its own rows there.
        "ctx_cache_batch_idx": gen_rows_out if self._ctx_block_tables is not None else slots,
        # Anchor token per gen request (block slot 0): last accepted
        # token. The dspark Markov chain conditions its first step on it.
        "first_prev_tokens": bonus,
        "ctx_kv_cache": self._ctx_kv_buf,
        # The manager-bound table is the read view rebased per slot above.
        "ctx_page_table": ctx_page_table,
    }
