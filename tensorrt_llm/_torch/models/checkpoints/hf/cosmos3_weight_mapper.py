# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from tensorrt_llm._torch.models.checkpoints.hf.qwen3vl_weight_mapper import (
    Qwen3VLHfWeightMapper,
)
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Cosmos3ForConditionalGeneration")
class Cosmos3HfWeightMapper(Qwen3VLHfWeightMapper):
    """
    Cosmos3 unified checkpoints store the Reasoner LLM with the old `model.` prefix
    stripped off and the ViT under flat `blocks.*` / `merger.*` / `patch_embed.*` /
    `pos_embed.*` / `deepstack_merger_list.*`. Re-target both to the nested Qwen3-VL
    layout (`model.language_model.*` and `model.visual.*`). Newer checkpoints also
    use Diffusers-style attention names; map them back to Qwen3-VL module names.
    """

    KEYS_TO_DROP = (
        # Generator (image / video diffusion) MoT expert + cross-modal projections
        r"\.add_q_proj\.",
        r"\.add_k_proj\.",
        r"\.add_v_proj\.",
        r"\.to_add_out\.",
        r"\.norm_added_q\.",
        r"\.norm_added_k\.",
        r"moe_gen",
        r"^proj_out\.",
        r"^proj_in\.",
        r"^time_embedder\.",
        # Sound tower
        r"^audio_proj_out\.",
        r"^audio_proj_in\.",
        r"^audio_modality_embed$",
        # Action tower
        r"^action_proj_out\.",
        r"^action_proj_in\.",
        r"^action_modality_embed$",
    )

    def __init__(self):
        super().__init__()

        self.prefix_params_map = {
            r"^(layers\.|embed_tokens\.|norm\.)": r"model.language_model.\1",
            r"^(blocks\.|merger\.|patch_embed\.|pos_embed\.|deepstack_merger_list\.)":
            r"model.visual.\1",
        }
        self.attn_params_map = {
            r"(.*)\.self_attn\.to_q\.(.*)": r"\1.self_attn.q_proj.\2",
            r"(.*)\.self_attn\.to_k\.(.*)": r"\1.self_attn.k_proj.\2",
            r"(.*)\.self_attn\.to_v\.(.*)": r"\1.self_attn.v_proj.\2",
            r"(.*)\.self_attn\.to_out\.(.*)": r"\1.self_attn.o_proj.\2",
            r"(.*)\.self_attn\.norm_q\.(.*)": r"\1.self_attn.q_norm.\2",
            r"(.*)\.self_attn\.norm_k\.(.*)": r"\1.self_attn.k_norm.\2",
        }

    @classmethod
    def should_drop_checkpoint_key(cls, key: str) -> bool:
        return any(re.search(pattern, key) for pattern in cls.KEYS_TO_DROP)

    def preprocess_weights(self, weights: dict) -> dict:
        weights = {
            key: value
            for key, value in weights.items()
            if not self.should_drop_checkpoint_key(key)
        }
        weights = self.rename_by_params_map(self.prefix_params_map, weights)
        return self.rename_by_params_map(self.attn_params_map, weights)
