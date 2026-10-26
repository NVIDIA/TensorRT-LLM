# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""In-tree configurations for StepFun Step-5 checkpoints.

A Step-5 checkpoint publishes a composite ``step5_vl`` configuration: a ``step5`` text
decoder under ``text_config`` and a ``perception_encoder`` vision tower under
``vision_config``. Neither ``transformers`` nor the checkpoint ships a configuration class,
so these classes let ``load_pretrained_config`` read the checkpoint without
``trust_remote_code``.

The text decoder's per-layer fields (``layer_types``, ``rope_theta``,
``partial_rotary_factors``, ``swiglu_limits``, ``swiglu_limits_shared``) carry one entry per
decoder layer followed by one per next-token-prediction (MTP) layer. The engine builds only
the ``num_hidden_layers`` decoder layers, so:

* ``layer_types`` holds the decoder layers alone, which is what the engine's generic
  per-layer window derivation and layer counts read; the trailing MTP entries are kept in
  ``mtp_layer_types``.
* the other per-layer lists are kept whole and are indexed by layer id; the accessors
  below (``rope_theta_for``, ``rotary_dim_for``, ...) do the indexing.

``num_attention_groups`` is the KV-head count and is also exposed under the name the engine
reads, ``num_key_value_heads``.
"""

from typing import Any, Optional, Union

from transformers.configuration_utils import PretrainedConfig

SLIDING_ATTENTION = "sliding_attention"
FULL_ATTENTION = "full_attention"


class Step5TextConfig(PretrainedConfig):
    """The Step-5 text decoder (``model_type`` ``step5``).

    Defaults are Step-5-Preview's published values for the shape fields. The per-layer
    lists have no default pattern: a config without ``layer_types`` describes a decoder
    whose layer kinds are unknown, and the accessors raise rather than guess.
    """

    model_type = "step5"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 128896,
        hidden_size: int = 4096,
        intermediate_size: int = 13824,
        num_hidden_layers: int = 92,
        num_attention_heads: int = 64,
        num_attention_groups: int = 4,
        head_dim: int = 192,
        rms_norm_eps: float = 1e-5,
        norm_dtype: str = "float32",
        fp32_residual_connection: bool = True,
        use_head_wise_attn_gate: bool = True,
        max_position_embeddings: int = 1048576,
        layer_types: Optional[list[str]] = None,
        sliding_window: Optional[int] = 512,
        rope_theta: Union[float, list[float]] = 10000.0,
        partial_rotary_factors: Optional[list[float]] = None,
        rope_scaling: Optional[dict[str, Any]] = None,
        yarn_only_types: Optional[list[str]] = None,
        use_moe: bool = True,
        moe_num_experts: int = 352,
        moe_top_k: int = 8,
        moe_intermediate_size: int = 1536,
        share_expert_dim: int = 1536,
        moe_layer_list: Optional[list[int]] = None,
        use_moe_router_bias: bool = True,
        need_fp32_gate: bool = True,
        norm_expert_weight: bool = True,
        moe_router_scaling_factor: float = 3.0,
        swiglu_limits: Optional[list[float]] = None,
        swiglu_limits_shared: Optional[list[float]] = None,
        num_nextn_predict_layers: int = 0,
        sparse_config: Optional[dict[str, Any]] = None,
        bos_token_id: int = 0,
        eos_token_id: Union[int, list[int]] = (1, 2),
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        kwargs.pop("model_type", None)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_key_value_heads = num_attention_groups
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.norm_dtype = norm_dtype
        self.fp32_residual_connection = fp32_residual_connection
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.yarn_only_types = list(yarn_only_types or [])
        self.num_nextn_predict_layers = num_nextn_predict_layers

        if layer_types is None:
            self.layer_types = None
            self.mtp_layer_types = []
        else:
            if len(layer_types) < num_hidden_layers:
                raise ValueError(
                    f"layer_types has {len(layer_types)} entries for "
                    f"{num_hidden_layers} decoder layers"
                )
            self.layer_types = list(layer_types[:num_hidden_layers])
            self.mtp_layer_types = list(layer_types[num_hidden_layers:])

        self.partial_rotary_factors = (
            None if partial_rotary_factors is None else list(partial_rotary_factors)
        )
        self.swiglu_limits = None if swiglu_limits is None else list(swiglu_limits)
        self.swiglu_limits_shared = (
            None if swiglu_limits_shared is None else list(swiglu_limits_shared)
        )
        for field, value in (
            ("partial_rotary_factors", self.partial_rotary_factors),
            ("swiglu_limits", self.swiglu_limits),
            ("swiglu_limits_shared", self.swiglu_limits_shared),
        ):
            if value is not None and len(value) < num_hidden_layers:
                raise ValueError(
                    f"{field} has {len(value)} entries for {num_hidden_layers} decoder layers"
                )

        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.num_experts_per_tok = kwargs.pop("num_experts_per_tok", moe_top_k)
        if self.num_experts_per_tok != moe_top_k:
            raise ValueError(
                f"num_experts_per_tok={self.num_experts_per_tok} disagrees with "
                f"moe_top_k={moe_top_k}"
            )
        self.moe_intermediate_size = moe_intermediate_size
        self.share_expert_dim = share_expert_dim
        self.moe_layer_list = list(moe_layer_list or [])
        self.use_moe_router_bias = use_moe_router_bias
        self.need_fp32_gate = need_fp32_gate
        self.norm_expert_weight = norm_expert_weight
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.sparse_config = None if sparse_config is None else dict(sparse_config)

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=list(eos_token_id)
            if isinstance(eos_token_id, (list, tuple))
            else eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Set after the base initializer, which would otherwise fold rope_theta (a list
        # here) into the rope_scaling dict while standardizing single-theta RoPE.
        if isinstance(rope_theta, (list, tuple)):
            if len(rope_theta) < num_hidden_layers:
                raise ValueError(
                    f"rope_theta has {len(rope_theta)} entries for "
                    f"{num_hidden_layers} decoder layers"
                )
            self.rope_theta = [float(theta) for theta in rope_theta]
        else:
            self.rope_theta = float(rope_theta)
        self.rope_scaling = None if rope_scaling is None else dict(rope_scaling)

    # --- per-layer accessors, indexed by layer id (MTP layers follow the decoder) ---

    def layer_type(self, layer_idx: int) -> str:
        """``sliding_attention`` or ``full_attention`` for a decoder or MTP layer."""
        if self.layer_types is None:
            raise ValueError("this Step-5 config carries no layer_types")
        if 0 <= layer_idx < self.num_hidden_layers:
            return self.layer_types[layer_idx]
        mtp_idx = layer_idx - self.num_hidden_layers
        if 0 <= mtp_idx < len(self.mtp_layer_types):
            return self.mtp_layer_types[mtp_idx]
        raise IndexError(
            f"layer {layer_idx} is outside the {self.num_hidden_layers} decoder and "
            f"{len(self.mtp_layer_types)} MTP layers"
        )

    def attention_window_for(self, layer_idx: int) -> Optional[int]:
        """The window a layer attends over: ``sliding_window`` on sliding layers, else None."""
        if self.layer_type(layer_idx) == SLIDING_ATTENTION:
            return self.sliding_window
        return None

    def is_sparse_layer(self, layer_idx: int) -> bool:
        """Whether the sparse indexer and sparse attention run on this layer."""
        sparse = self.sparse_config
        if not sparse or not sparse.get("enabled", True):
            return False
        return self.layer_type(layer_idx) in sparse.get("apply_to_layer_types", [FULL_ATTENTION])

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Whether this layer's MLP is the routed-expert block (else a dense SwiGLU)."""
        return self.use_moe and layer_idx in self.moe_layer_list

    def rope_theta_for(self, layer_idx: int) -> float:
        if isinstance(self.rope_theta, list):
            return self.rope_theta[layer_idx]
        return self.rope_theta

    def rotary_dim_for(self, layer_idx: int) -> int:
        """Rotated channels of a layer's ``head_dim``: ``head_dim * partial_rotary_factor``."""
        factor = (
            1.0 if self.partial_rotary_factors is None else self.partial_rotary_factors[layer_idx]
        )
        return int(round(self.head_dim * factor))

    def rope_scaling_for(self, layer_idx: int) -> Optional[dict[str, Any]]:
        """``rope_scaling`` where it applies: on the layer types ``yarn_only_types`` names,
        or on every layer when that list is empty."""
        if self.rope_scaling is None:
            return None
        if self.yarn_only_types and self.layer_type(layer_idx) not in self.yarn_only_types:
            return None
        return self.rope_scaling

    def swiglu_limit_for(self, layer_idx: int) -> Optional[float]:
        """The routed (or dense) SwiGLU clamp of a layer."""
        return None if self.swiglu_limits is None else self.swiglu_limits[layer_idx]

    def swiglu_limit_shared_for(self, layer_idx: int) -> Optional[float]:
        """The shared expert's SwiGLU clamp of a layer."""
        return None if self.swiglu_limits_shared is None else self.swiglu_limits_shared[layer_idx]

    def to_dict(self) -> dict[str, Any]:
        """Serialize in the checkpoint's own shape: one ``layer_types`` list over the
        decoder and MTP layers, and ``rope_scaling`` under its published name."""
        output = super().to_dict()
        mtp_layer_types = output.pop("mtp_layer_types", None) or []
        if output.get("layer_types") is not None:
            output["layer_types"] = list(output["layer_types"]) + list(mtp_layer_types)
        if "rope_parameters" in output:
            output["rope_scaling"] = output.pop("rope_parameters")
        return output


class Step5VisionConfig(PretrainedConfig):
    """The Step-5 perception-encoder vision tower (``model_type`` ``perception_encoder``)."""

    model_type = "perception_encoder"
    base_config_key = "vision_config"

    def __init__(
        self,
        image_size: int = 728,
        patch_size: int = 14,
        width: int = 1536,
        layers: int = 47,
        heads: int = 16,
        hidden_act: str = "quick_gelu",
        pool_type: str = "none",
        output_dim: Optional[int] = None,
        use_cls_token: bool = False,
        ls_init_value: Optional[float] = 0.1,
        use_ln_post: bool = False,
        **kwargs,
    ) -> None:
        kwargs.pop("model_type", None)
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.hidden_act = hidden_act
        self.pool_type = pool_type
        self.output_dim = output_dim
        self.use_cls_token = use_cls_token
        self.ls_init_value = ls_init_value
        self.use_ln_post = use_ln_post
        super().__init__(**kwargs)


class Step5Config(PretrainedConfig):
    """The published composite (``model_type`` ``step5_vl``): text decoder plus vision tower."""

    model_type = "step5_vl"
    sub_configs = {
        "text_config": Step5TextConfig,
        "vision_config": Step5VisionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: Optional[Union[dict, Step5TextConfig]] = None,
        vision_config: Optional[Union[dict, Step5VisionConfig]] = None,
        image_token_id: int = 128001,
        image_token_len: int = 169,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        vision_select_layer: int = -1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        kwargs.pop("model_type", None)
        if isinstance(text_config, dict):
            text_config = Step5TextConfig(**text_config)
        elif text_config is None:
            text_config = Step5TextConfig()
        if isinstance(vision_config, dict):
            vision_config = Step5VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Step5VisionConfig()
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.image_token_len = image_token_len
        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.vision_select_layer = vision_select_layer
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
