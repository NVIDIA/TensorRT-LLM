# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 configuration compatibility for Transformers versions before native support.

Image/video preprocessing still requires the native Glm5NextProcessor.
Native Transformers config classes take precedence when available.
"""

from transformers import PretrainedConfig

from .deepseek_v3 import DeepseekV3Config


class Glm5NextTextConfig(DeepseekV3Config):
    model_type = "glm5_next_text"
    base_config_key = "text_config"
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(self, **kwargs):
        defaults = dict(
            vocab_size=154880,
            hidden_size=4096,
            intermediate_size=12288,
            moe_intermediate_size=2048,
            num_hidden_layers=45,
            num_attention_heads=64,
            num_key_value_heads=64,
            n_routed_experts=288,
            qk_rope_head_dim=0,
            qk_nope_head_dim=256,
            v_head_dim=256,
            n_group=1,
            max_position_embeddings=1048576,
            rms_norm_eps=1e-5,
            pad_token_id=154820,
            bos_token_id=None,
            eos_token_id=None,
            index_topk=2048,
            index_head_dim=128,
            index_n_heads=32,
            index_kpool=16,
            index_kpool_always_select_tail=True,
            swiglu_limit=10.0,
            hc_mult=4,
            hc_eps=1e-6,
            hc_sinkhorn_iters=20,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
        )
        fields = {**defaults, **kwargs}
        # Older Transformers rejects deepseek_sparse_attention in its generic validator.
        layer_types = fields.pop("layer_types", None)
        super().__init__(**fields)
        self.head_dim = self.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.layer_types = [
            "deepseek_sparse_attention" if kind == "full_attention" else kind
            for kind in (
                layer_types
                if layer_types is not None
                else [
                    "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
                    for i in range(self.num_hidden_layers)
                ]
            )
        ]
        if getattr(self, "mlp_layer_types", None) is None:
            self.mlp_layer_types = [
                "dense" if i < 3 else "sparse" for i in range(self.num_hidden_layers)
            ]
        if getattr(self, "indexer_types", None) is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern]
                    if isinstance(pattern, str)
                    else list(pattern)
                )
            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if max(i - offset + 1, 0) % freq == 0 else "shared"
                    for i in range(self.num_hidden_layers)
                ]
        linear = kwargs.get("linear_attn_config") or {}
        self.linear_head_dim = linear.get("head_dim", kwargs.get("linear_head_dim", 128))
        self.linear_num_heads = linear.get("num_heads", kwargs.get("linear_num_heads", 64))
        self.linear_conv_kernel_dim = linear.get(
            "short_conv_kernel_size", kwargs.get("linear_conv_kernel_dim", 4)
        )
        self.linear_lower_bound = linear.get(
            "gate_lower_bound", kwargs.get("linear_lower_bound", -5.0)
        )
        if (
            kwargs.get("linear_attn_config") is not None
            and linear.get("safe_gate", True)
            and self.linear_lower_bound is None
        ):
            self.linear_lower_bound = -5.0


class Glm5NextVisionConfig(PretrainedConfig):
    model_type = "glm5_next_vision"
    base_config_key = "vision_config"
    attribute_map = {"num_attention_heads": "num_heads"}

    def __init__(self, **kwargs):
        defaults = dict(
            depth=24,
            hidden_size=1024,
            hidden_act="silu",
            attention_bias=True,
            attention_dropout=0.0,
            num_heads=16,
            in_channels=3,
            image_size=336,
            patch_size=14,
            rms_norm_eps=1e-5,
            spatial_merge_size=2,
            temporal_patch_size=2,
            out_hidden_size=1536,
            intermediate_size=4096,
            initializer_range=0.02,
            projection_intermediate_size=10240,
            swiglu_limit=10.0,
        )
        super().__init__(**{**defaults, **kwargs})


class Glm5NextConfig(PretrainedConfig):
    model_type = "glm5_next"
    sub_configs = {"text_config": Glm5NextTextConfig, "vision_config": Glm5NextVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        self.text_config = (
            Glm5NextTextConfig(**(text_config if text_config is not None else kwargs))
            if isinstance(text_config, dict) or text_config is None
            else text_config
        )
        self.vision_config = (
            Glm5NextVisionConfig(**(vision_config or {}))
            if isinstance(vision_config, dict) or vision_config is None
            else vision_config
        )
        defaults = dict(
            image_token_id=154854,
            video_token_id=154855,
            image_start_token_id=154830,
            image_end_token_id=154831,
            video_start_token_id=154832,
            video_end_token_id=154833,
            tie_word_embeddings=False,
        )
        super().__init__(**{**defaults, **kwargs})
