# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-tree configurations for Qwen3.8-Flash-Next checkpoints.

The checkpoint's ``text_config`` describes a hybrid decoder with Gated-DeltaNet
linear-attention layers, QSA full-attention layers, routed and shared experts,
Hyper-Connection residual streams, and a PLE n-gram side path.

``Qwen4ExpTextConfig`` subclasses ``transformers.Qwen3NextConfig``: the qwen4_exp
text core shares Qwen3Next's hybrid layer schema (``layer_types`` /
``full_attention_interval``, the ``linear_*`` GDN dims, and the MoE
``num_experts`` / ``moe_intermediate_size`` fields), so inheriting Qwen3Next's
field set keeps the text config construction-compatible with the shared Qwen3Next
hybrid runtime and ``KVCacheManagerV2`` while the qwen4_exp-only fields
(Hyper-Connection ``hc_*``, PLE ``ple_*`` / ``ngram_*``, QSA ``indexer_*``, the
attention ``output_gate_type``) are carried as extra attributes for the qwen4_exp
model to read.

``pyexecutor.config_utils.load_pretrained_config`` can flatten the checkpoint's
composite configuration to this text configuration for language-only serving.
"""

from typing import Optional, Union

from transformers import PretrainedConfig, Qwen3NextConfig


def _flatten_qwen4_exp_rope(fields: dict) -> None:
    """Flatten a nested ``rope_parameters`` block into top-level rope fields.

    Qwen4-Exp nests rope metadata under ``rope_parameters`` (``rope_theta``,
    ``partial_rotary_factor``, and the MRoPE ``mrope_section`` /
    ``mrope_interleaved``). The shared Qwen3Next runtime and ``RopeParams`` read
    ``rope_theta``, ``partial_rotary_factor``, and ``rope_scaling`` (with
    ``type="mrope"`` when MRoPE is present) as top-level fields. Mutates
    ``fields`` in place; a no-op when ``rope_parameters`` is absent (e.g. a
    hand-built reduced test config that already sets the flat fields).
    """
    rope_parameters = dict(fields.pop("rope_parameters", None) or {})
    if not rope_parameters:
        return
    rope_theta = rope_parameters.pop("rope_theta", None)
    if rope_theta is not None:
        fields.setdefault("rope_theta", rope_theta)
    partial_rotary_factor = rope_parameters.pop("partial_rotary_factor", None)
    if partial_rotary_factor is not None:
        fields.setdefault("partial_rotary_factor", partial_rotary_factor)
    # Merge any remaining rope_parameters (mrope_section/mrope_interleaved/
    # rope_type) with an explicit rope_scaling, preferring rope_scaling.
    rope_scaling = {**rope_parameters, **(dict(fields.get("rope_scaling") or {}))}
    if not rope_scaling:
        return
    has_mrope = "mrope_section" in rope_scaling or rope_scaling.get("mrope_interleaved", False)
    if has_mrope:
        rope_scaling["type"] = "mrope"
        rope_scaling.pop("rope_type", None)
        fields["rope_scaling"] = rope_scaling
    else:
        rope_type = rope_scaling.pop("rope_type", None)
        # "default" == standard RoPE (no scaling); leave rope_scaling unset so
        # downstream code does not take a scaling path.
        if rope_type not in (None, "default"):
            rope_scaling["type"] = rope_type
            fields["rope_scaling"] = rope_scaling


def _normalize_qwen4_exp_layer_types(fields: dict) -> None:
    """Map the HF QSA layer alias to TRT-LLM's hybrid-layer label."""
    layer_types = fields.get("layer_types")
    if layer_types is None:
        return
    fields["layer_types"] = [
        "full_attention" if layer_type == "deepseek_sparse_attention" else layer_type
        for layer_type in layer_types
    ]


class Qwen4ExpTextConfig(Qwen3NextConfig):
    """Text (language-model) config for Qwen4-Exp (``qwen4_exp_text``).

    Subclasses :class:`transformers.Qwen3NextConfig` because the qwen4_exp text
    core shares the Qwen3Next hybrid layer schema, so the shared Qwen3Next
    runtime + hybrid KV cache construct against it unchanged. The
    qwen4_exp-specific fields are carried as extra attributes:

      * Hyper-Connection residual: ``hc_count``, ``hc_lowrank``
      * PLE n-gram short-conv side path: ``ple_layer_ids``, ``ple_embed_dim``,
        ``ple_conv_kernel_size``, ``ngram_size``, ``heads_per_ngram``,
        ``ngram_vocab_size_base``, ``make_ngram_vocab_size_divisible_by``,
        ``split_ngram_parts``
      * QSA compressed sparse indexer: ``indexer_n_heads``,
        ``indexer_kv_heads``, ``indexer_head_dim``, ``indexer_budget``,
        ``indexer_compress_ratio``
      * attention output gate: ``output_gate_type``
    """

    model_type = "qwen4_exp_text"

    def __init__(self, **kwargs):
        _flatten_qwen4_exp_rope(kwargs)
        _normalize_qwen4_exp_layer_types(kwargs)
        # qwen4_exp is MoE-only and ships no dense ``intermediate_size``;
        # synthesize the Qwen3Next alias (top-k routed experts + shared expert)
        # so any ``intermediate_size`` read during construction is well-defined.
        if kwargs.get("intermediate_size") is None:
            moe_inter = kwargs.get("moe_intermediate_size")
            top_k = kwargs.get("num_experts_per_tok")
            shared = kwargs.get("shared_expert_intermediate_size") or 0
            if moe_inter is not None and top_k is not None:
                kwargs["intermediate_size"] = top_k * moe_inter + shared
        super().__init__(**kwargs)


class Qwen4ExpVisionConfig(PretrainedConfig):
    """Configuration for the Qwen3-VL-compatible vision tower."""

    model_type = "qwen4_exp_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 27,
        hidden_size: int = 1152,
        hidden_act: str = "gelu_pytorch_tanh",
        intermediate_size: int = 4304,
        num_heads: int = 16,
        in_channels: int = 3,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 2560,
        num_position_embeddings: int = 2304,
        deepstack_visual_indexes: Optional[list[int]] = None,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.deepstack_visual_indexes = list(deepstack_visual_indexes or [])
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class Qwen4ExpConfig(PretrainedConfig):
    """Composite text-and-vision configuration."""

    model_type = "qwen4_exp"
    sub_configs = {
        "text_config": Qwen4ExpTextConfig,
        "vision_config": Qwen4ExpVisionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: Optional[Union[dict, Qwen4ExpTextConfig]] = None,
        vision_config: Optional[Union[dict, Qwen4ExpVisionConfig]] = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = Qwen4ExpTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen4ExpTextConfig()
        if isinstance(vision_config, dict):
            # Early checkpoints used the composite model type for this nested
            # block. The runtime class is unambiguous once it is nested here.
            vision_config = dict(vision_config)
            if vision_config.get("model_type") == "qwen4_exp":
                vision_config["model_type"] = "qwen4_exp_vision"
            vision_config = Qwen4ExpVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen4ExpVisionConfig()

        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
