# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MTP step-zero tail selection with the shared worker's full-row output contract."""

import torch

from tensorrt_llm._torch.speculative import MTPEagleWorker

from .ops.spec.mtp_tail import scatter_mtp_tail


class RubinMTPEagleWorker(MTPEagleWorker):
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
