# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Cosmos3EdgeForConditionalGeneration")
class Cosmos3EdgeHfWeightMapper(HfWeightMapper):
    """Map a unified Cosmos3-Edge checkpoint onto the Reasoner modules.

    Text weights live under ``transformer/`` without a ``model.`` prefix and use
    Diffusers attention names (``to_q`` / ``to_k`` / ``to_v`` / ``to_out``).
    Vision and projector weights already use ``model.visual`` / ``model.projector``.
    Generator-only tensors in the same transformer shards are dropped.
    """

    KEYS_TO_DROP = (
        r"_moe_gen",
        r"k_norm_und_for_gen",
        r"\.add_q_proj\.",
        r"\.add_k_proj\.",
        r"\.add_v_proj\.",
        r"\.to_add_out\.",
        r"\.norm_added_q\.",
        r"\.norm_added_k\.",
        r"(^|\.)proj_out\.",
        r"(^|\.)proj_in\.",
        r"(^|\.)time_embedder\.",
        r"(^|\.)action_proj_out\.",
        r"(^|\.)action_proj_in\.",
        r"(^|\.)action_modality_embed",
        r"(^|\.)audio_proj_out\.",
        r"(^|\.)audio_proj_in\.",
        r"(^|\.)audio_modality_embed",
    )

    def __init__(self):
        super().__init__()
        self.attn_params_map = {
            r"(.*)\.self_attn\.to_q\.(.*)": r"\1.self_attn.q_proj.\2",
            r"(.*)\.self_attn\.to_k\.(.*)": r"\1.self_attn.k_proj.\2",
            r"(.*)\.self_attn\.to_v\.(.*)": r"\1.self_attn.v_proj.\2",
            r"(.*)\.self_attn\.to_out\.(.*)": r"\1.self_attn.o_proj.\2",
            r"(.*)\.self_attn\.out_proj\.(.*)": r"\1.self_attn.o_proj.\2",
        }
        self.mlp_params_map = {
            r"(.*)\.mlp\.fc1\.(.*)": r"\1.mlp.up_proj.\2",
            r"(.*)\.mlp\.fc2\.(.*)": r"\1.mlp.down_proj.\2",
        }
        # Bare text keys only. Vision keys already start with ``model.visual``.
        self.prefix_params_map = {
            r"^(layers\.|embed_tokens\.|norm\.)": r"model.\1",
        }

    @classmethod
    def should_drop_checkpoint_key(cls, key: str) -> bool:
        return any(re.search(pattern, key) for pattern in cls.KEYS_TO_DROP)

    def preprocess_weights(self, weights: dict) -> dict:
        weights = {
            key: value for key, value in weights.items() if not self.should_drop_checkpoint_key(key)
        }
        weights = self.rename_by_params_map(self.attn_params_map, weights)
        weights = self.rename_by_params_map(self.mlp_params_map, weights)
        return self.rename_by_params_map(self.prefix_params_map, weights)
