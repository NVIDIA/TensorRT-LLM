# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The repeated prediction layer and shared vocabulary used by the one-model MTP worker."""

import torch
from torch.nn import functional as F

from .attention_metadata import attention_step_args
from .decoder import rms_norm


class DraftLayer:
    def __init__(self, core, logits_processor):
        self.core = core
        self.shared_head = logits_processor

    def __call__(
        self,
        *,
        embed_tokens,
        all_rank_num_tokens,
        input_ids,
        hidden_states,
        attn_metadata,
        **kwargs,
    ):
        core = self.core
        weights = core.mtp
        if hidden_states.shape[0] == 0:
            if core.attn_dp and any(all_rank_num_tokens):
                core.moe(weights, hidden_states, hidden_states, all_rank_num_tokens, core.tp_reduce)
            return hidden_states

        embedding = F.embedding(input_ids.long(), embed_tokens)
        x = core.mm(
            torch.cat(
                (
                    rms_norm(embedding, weights["enorm"], core.eps),
                    rms_norm(hidden_states, weights["hnorm"], core.eps),
                ),
                dim=-1,
            ),
            weights["eh"],
        )

        md = attn_metadata
        layer = core.num_layers
        # The worker switches the active KV manager between the target and draft passes.
        local_layer = md.kv_cache_manager.layer_offsets[layer]
        step = attention_step_args(md, decode=md.num_contexts == 0, n_rows=x.shape[0])
        plan = core.index.plan(md, x.shape[0]) if core.index is not None else None
        x = x + core.attention(
            local_layer,
            weights,
            rms_norm(x, weights["norm_in"], core.eps),
            step,
            md,
            core.tp_reduce,
            plan,
            layer,
        )
        # Both expert paths consume the post-attention state of this ordinary residual block.
        moe_input = rms_norm(x, weights["pre_moe_norm"], core.eps)
        x = x + core.moe(weights, moe_input, moe_input, all_rank_num_tokens, core.tp_reduce)
        return rms_norm(x, weights["final_norm"], core.eps)


class DraftModel:
    """Expose the repeated layer and the target's embedding and head to MTPEagleWorker."""

    def __init__(self, core, lm_head, logits_processor):
        self.core = core
        self.lm_head = lm_head
        self.mtp_layers = [DraftLayer(core, logits_processor)]

    @property
    def embed_tokens(self):
        return self.core.w["embed"]
