# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned ADP MTP loop with compact tails and a fused greedy draft head."""

import contextlib

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.sparse.dsa import is_dsa_cache_manager
from tensorrt_llm._torch.attention.backends.sparse.params import MTPIndexShareMetadata
from tensorrt_llm._torch.speculative import MTPEagleWorker
from tensorrt_llm._torch.speculative.eagle3 import _reset_mtp_index_share
from tensorrt_llm._torch.utils import create_lm_head_tp_mapping

from .ops.spec.draft_sample_glue import DraftSampleGlueAdp
from .ops.spec.mtp_tail import scatter_mtp_tail


class RubinMTPEagleWorker(MTPEagleWorker):
    def _adp_plan(
        self, inputs, attn_metadata, spec_metadata, draft_model, num_accepted_tokens
    ) -> DraftSampleGlueAdp | None:
        """Choose the same collective path on every rank, before any workspace split."""
        counts = spec_metadata.all_rank_num_gens
        positions = inputs["position_ids"]
        mapping = self.model_config.mapping
        # Counts include CUDA-graph padding and are shared across the EP world.
        # Mixed prefill and empty ranks use the shared loop on every rank.
        if (
            not counts
            or spec_metadata.all_rank_num_seqs != counts
            or spec_metadata.subseq_all_rank_num_tokens != counts
            or min(counts) <= 0
            or max(counts) > spec_metadata.max_num_requests
            or not mapping.enable_lm_head_tp_in_adp
            or not isinstance(attn_metadata, MTPIndexShareMetadata)
            or not is_dsa_cache_manager(attn_metadata.kv_cache_manager)
            or self.guided_decoder is not None
            or self._d2t is not None
            or self.sa_enhancer is not None
            or spec_metadata.is_layer_capture(0)
            or positions.dim() != 1
            or positions.dtype not in (torch.int32, torch.int64)
            or not positions.is_contiguous()
            or num_accepted_tokens.dtype != torch.int32
            or not num_accepted_tokens.is_contiguous()
            or torch.compiler.is_compiling()
        ):
            return None
        head = draft_model.lm_head
        head_mapping = create_lm_head_tp_mapping(mapping, spec_metadata.max_num_requests)
        if (
            head.has_any_quant
            or head.weight.dtype != torch.bfloat16
            or head.output_dtype not in (None, torch.bfloat16)
            or head_mapping.tp_size < 2
            or head.out_features % (8 * head_mapping.tp_size) != 0
        ):
            return None
        plan = DraftSampleGlueAdp.get(
            head_mapping, spec_metadata.max_num_requests, head.in_features
        )
        # This flag is synchronized within an LM-head group, not necessarily
        # across the world. All groups must finish the workspace splits first.
        return None if spec_metadata.wants_advanced_draft_sampling else plan

    def _forward_linear_draft_loop(
        self,
        inputs,
        attn_metadata,
        spec_metadata,
        draft_model,
        draft_kv_cache_manager,
        num_contexts,
        batch_size,
        num_accepted_tokens,
        original_all_rank_num_tokens,
        accepted_tokens=None,
        pre_draft_seq_lens=None,
    ):
        plan = self._adp_plan(
            inputs, attn_metadata, spec_metadata, draft_model, num_accepted_tokens
        )
        if plan is None:
            return super()._forward_linear_draft_loop(
                inputs,
                attn_metadata,
                spec_metadata,
                draft_model,
                draft_kv_cache_manager,
                num_contexts,
                batch_size,
                num_accepted_tokens,
                original_all_rank_num_tokens,
                accepted_tokens,
                pre_draft_seq_lens,
            )

        draft_len = spec_metadata.runtime_draft_len
        layer = draft_model.mtp_layers[0]
        glue = layer.draft_sample_glue_window(
            draft_model.embed_tokens, inputs["position_ids"].device
        )
        draft_tokens = torch.empty(
            (batch_size, draft_len), dtype=torch.int32, device=inputs["position_ids"].device
        )
        batch_indices = spec_metadata.batch_indices_cuda[:batch_size]
        row_ids = batch_indices.long() * (draft_len + 1) + num_accepted_tokens[:batch_size] - 1
        next_new_tokens = None
        if (
            accepted_tokens is not None
            and accepted_tokens.dtype == torch.int32
            and accepted_tokens.dim() == 2
            and accepted_tokens.stride(1) == 1
            and accepted_tokens.shape[1] >= draft_len + 1
            and batch_indices.dtype == torch.int32
            and batch_indices.is_contiguous()
        ):
            next_new_tokens = draft_tokens.new_empty((batch_size, draft_len + 1))

        with contextlib.ExitStack() as scope:
            attn_metadata.set_in_mtp_draft_loop(True)
            attn_metadata.set_skip_topk(False)
            attn_metadata.set_mtp_num_accepted(num_accepted_tokens)
            scope.callback(_reset_mtp_index_share, attn_metadata)
            metadata = scope.enter_context(
                self.draft_kv_cache_context(attn_metadata, draft_kv_cache_manager)
            )
            inputs["attn_metadata"] = metadata
            # The separate draft cache must see the corrected overlap-scheduling lengths.
            if is_dsa_cache_manager(draft_kv_cache_manager):
                metadata.on_update_kv_lens()
            for step in range(draft_len):
                metadata.set_skip_topk(step > 0)
                # Attention still handles all accepted-prefix rows on step zero.
                # The MTP layer selects the tail only after attention, before MoE.
                hidden = layer(
                    embed_tokens=draft_model.embed_tokens,
                    all_rank_num_tokens=self._get_step_all_rank_num_tokens(spec_metadata, step),
                    tail_row_ids=row_ids if step == 0 else None,
                    tail_all_rank_num_tokens=spec_metadata.subseq_all_rank_num_tokens,
                    **inputs,
                )
                padding = plan.segment_rows - batch_size
                head_hidden = F.pad(hidden, (0, 0, 0, padding)) if padding else hidden
                logits = layer.shared_head(
                    head_hidden,
                    draft_model.lm_head,
                    metadata,
                    True,
                    gather=plan.gather,
                )
                last = step + 1 == draft_len
                positions = None if last else inputs["position_ids"].new_empty((batch_size,))
                glue_out = (
                    hidden.new_empty((batch_size, glue.col_hi - glue.col_lo))
                    if glue is not None and not last
                    else None
                )
                plan.run(
                    logits,
                    draft_tokens,
                    step,
                    glue=glue,
                    glue_out=glue_out,
                    hidden=hidden,
                    positions_in=inputs["position_ids"],
                    positions_out=positions,
                    gather_ids=row_ids if step == 0 else None,
                    seq_lens=metadata.seq_lens_cuda[:batch_size],
                    kv_lens=metadata.kv_lens_cuda[:batch_size],
                    num_accepted=num_accepted_tokens[:batch_size],
                    accepted_tokens=accepted_tokens
                    if last and next_new_tokens is not None
                    else None,
                    batch_indices=batch_indices if last and next_new_tokens is not None else None,
                    next_new_tokens=next_new_tokens if last else None,
                )
                if step == 0:
                    metadata._seq_lens[:batch_size].fill_(1)
                    metadata.on_update()
                    metadata.num_contexts = 0
                    self._prepare_flash_mla_generation_layout(metadata, 0, batch_size)
                # Refresh attention metadata after every draft token so cache mappings
                # and lengths match the next step, including when TopK is reused.
                metadata.update_for_spec_dec()
                if step == 0:
                    metadata.use_spec_decoding = False
                if not last:
                    inputs = {
                        "input_ids": draft_tokens[:, step],
                        "position_ids": positions,
                        "hidden_states": hidden,
                        "attn_metadata": metadata,
                        "spec_metadata": spec_metadata,
                        "eh_proj_input": glue_out,
                    }
        return draft_tokens, next_new_tokens

    def _run_draft_forward(self, draft_model, inputs, spec_metadata, step_idx: int):
        metadata = inputs["attn_metadata"]
        all_rank_num_gens = spec_metadata.all_rank_num_gens
        tail_counts = spec_metadata.subseq_all_rank_num_tokens
        # Every EP rank must use the same MoE row counts. The shared input
        # preparation exchanges these lists, including CUDA-graph dummy rows.
        # Mixed-context or empty-rank batches retain the original full-row path.
        if (
            step_idx != 0
            or not all_rank_num_gens
            or tail_counts != all_rank_num_gens
            or not all(count > 0 for count in all_rank_num_gens)
            or spec_metadata.is_layer_capture(0)
        ):
            return super()._run_draft_forward(draft_model, inputs, spec_metadata, step_idx)
        # The DSA draft loop sets accepted counts before invoking this method.
        accepted = metadata.mtp_num_accepted
        assert accepted is not None
        requests = metadata.num_generations
        row_ids = (
            spec_metadata.batch_indices_cuda[:requests].to(torch.long)
            * (spec_metadata.runtime_draft_len + 1)
            + accepted[:requests].to(torch.long)
            - 1
        )
        selected = draft_model.mtp_layers[0](
            embed_tokens=draft_model.embed_tokens,
            all_rank_num_tokens=self._get_step_all_rank_num_tokens(spec_metadata, step_idx),
            tail_row_ids=row_ids,
            tail_all_rank_num_tokens=tail_counts,
            **inputs,
        )
        return scatter_mtp_tail(selected, row_ids, inputs["hidden_states"].shape[0]), None
