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
"""Architecture configuration for Beam checkpoints."""

from os import PathLike
from typing import Any, Optional, Union

import torch
from huggingface_hub.dataclasses import strict
from transformers import PretrainedConfig

BEAM_LAYER_TYPES = ("sliding_attention", "full_attention")
BEAM_MLP_LAYER_TYPES = ("dense", "sparse")
BEAM_ROPE_TYPES = ("default", "linear")


def _default_layer_types(num_hidden_layers: int) -> list[str]:
    layer_types = [
        "sliding_attention" if (i + 1) % 6 else "full_attention" for i in range(num_hidden_layers)
    ]
    if layer_types:
        layer_types[-1] = "full_attention"
    return layer_types


def _default_mlp_layer_types(num_hidden_layers: int) -> list[str]:
    num_dense = min(3, num_hidden_layers)
    return ["dense"] * num_dense + ["sparse"] * (num_hidden_layers - num_dense)


def _normalize_rope_parameters(raw: Optional[dict]) -> dict:
    """Validate Beam's sliding-RoPE / full-NoPE rope parameters."""
    if raw is None:
        return {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        }
    if not isinstance(raw, dict) or set(raw) != {"sliding_attention", "full_attention"}:
        raise ValueError(
            "rope_parameters must contain exactly 'sliding_attention' and 'full_attention': "
            f"{raw!r}"
        )
    if raw["full_attention"] is not None:
        raise ValueError("Beam full-attention layers run without rotary embeddings")
    sliding = raw["sliding_attention"]
    if not isinstance(sliding, dict):
        raise ValueError("sliding_attention rope parameters must be a dictionary")
    rope_type = sliding.get("rope_type", "default")
    if rope_type not in BEAM_ROPE_TYPES:
        raise ValueError(f"rope_type must be one of {BEAM_ROPE_TYPES}: {rope_type!r}")
    theta = sliding.get("rope_theta", 10000.0)
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise ValueError(f"rope_theta must be a number: {theta!r}")
    normalized = {"rope_type": rope_type, "rope_theta": float(theta)}
    if rope_type == "linear":
        factor = sliding.get("factor")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor <= 0:
            raise ValueError(f"linear RoPE needs a positive factor: {factor!r}")
        normalized["factor"] = float(factor)
    return {"sliding_attention": normalized, "full_attention": None}


@strict
class BeamConfig(PretrainedConfig):
    """Architecture configuration of a Beam checkpoint (``config.json`` fields)."""

    model_type = "beam"
    # Preserve full config equality for Beam fields set in __init__.
    __eq__ = PretrainedConfig.__eq__

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 5120,
        intermediate_size: int = 20480,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-6,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: Optional[int] = 512,
        moe_intermediate_size: int = 2560,
        num_experts_per_tok: int = 8,
        num_experts: int = 256,
        norm_topk_prob: bool = True,
        n_group: int = 8,
        topk_group: int = 4,
        routed_scaling_factor: float = 2.5,
        final_logit_softcapping: Optional[float] = 30.0,
        layer_types: Optional[list[str]] = None,
        mlp_layer_types: Optional[list[str]] = None,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype = dtype if dtype is not None else torch_dtype
        if dtype is None:
            dtype = torch.bfloat16
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        kwargs.setdefault("architectures", ["BeamForCausalLM"])
        super().__init__(tie_word_embeddings=tie_word_embeddings, dtype=dtype, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_parameters = _normalize_rope_parameters(rope_parameters)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.final_logit_softcapping = final_logit_softcapping
        self.tie_word_embeddings = tie_word_embeddings
        self.layer_types = (
            list(layer_types) if layer_types else _default_layer_types(num_hidden_layers)
        )
        self.mlp_layer_types = (
            list(mlp_layer_types)
            if mlp_layer_types
            else _default_mlp_layer_types(num_hidden_layers)
        )
        if getattr(self, "torch_dtype", None) != dtype:
            try:
                self.torch_dtype = dtype
            except AttributeError:
                # A read-only torch_dtype property derives from ``dtype``.
                self.dtype = dtype
        if self.torch_dtype != dtype:
            raise ValueError(f"BeamConfig dtype {self.torch_dtype!r} != requested {dtype!r}")
        self._validate()

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, PathLike], **kwargs: Any
    ) -> Union["BeamConfig", tuple["BeamConfig", dict[str, Any]]]:
        # Direct config-class loading must retain the checkpoint location, just
        # as AutoConfig does, so the model loader can find the weight shards.
        kwargs.setdefault("name_or_path", str(pretrained_model_name_or_path))
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def validate_rope(self) -> None:
        """Validate the per-layer settings at save time, including full attention without RoPE."""
        _normalize_rope_parameters(self.rope_parameters)

    def _validate(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"len(layer_types)={len(self.layer_types)} must equal "
                f"num_hidden_layers={self.num_hidden_layers}"
            )
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"len(mlp_layer_types)={len(self.mlp_layer_types)} must equal "
                f"num_hidden_layers={self.num_hidden_layers}"
            )
        unsupported = set(self.layer_types).difference(BEAM_LAYER_TYPES)
        if unsupported:
            raise ValueError(f"unsupported layer_types: {sorted(unsupported)}")
        unsupported = set(self.mlp_layer_types).difference(BEAM_MLP_LAYER_TYPES)
        if unsupported:
            raise ValueError(f"unsupported mlp_layer_types: {sorted(unsupported)}")
        if self.hidden_act != "silu":
            raise ValueError(f"hidden_act must be 'silu': {self.hidden_act!r}")
        if self.num_key_value_heads == 0 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"num_attention_heads={self.num_attention_heads} must be divisible by "
                f"a nonzero num_key_value_heads, got {self.num_key_value_heads}"
            )
        if "sliding_attention" in self.layer_types and not self.sliding_window:
            raise ValueError(
                f"sliding_attention layers need a positive sliding_window, got {self.sliding_window!r}"
            )
        if "sparse" in self.mlp_layer_types:
            if self.n_group == 0 or self.num_experts % self.n_group:
                raise ValueError(
                    f"num_experts={self.num_experts} must be divisible by a nonzero "
                    f"n_group, got {self.n_group}"
                )
            if not 1 <= self.topk_group <= self.n_group:
                raise ValueError(
                    f"topk_group={self.topk_group} must be in [1, n_group={self.n_group}]"
                )
            experts_per_group = self.num_experts // self.n_group
            if experts_per_group < 2:
                raise ValueError(
                    f"Beam group scores need at least 2 experts per group, got {experts_per_group} "
                    f"(num_experts={self.num_experts}, n_group={self.n_group})"
                )
            selected_experts = self.topk_group * experts_per_group
            if self.num_experts_per_tok > selected_experts:
                raise ValueError(
                    f"num_experts_per_tok={self.num_experts_per_tok} exceeds the {selected_experts} "
                    f"available experts (topk_group={self.topk_group}, "
                    f"num_experts={self.num_experts}, n_group={self.n_group})"
                )
