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
"""Config for the MuseSparkOpen omni checkpoint, text side only.

The checkpoint's ``config.json`` nests most decoder knobs under ``moe``,
``metap`` and ``numerics_args`` and describes the attention schedule with
``global_attn_cfg`` + ``every_n_layers_nope`` rather than the flat
``layer_types`` / ``sliding_window`` pair the runtime reads. This class keeps
the raw fields and additionally publishes the flat ones so the generic
KV-cache window derivation, MoE shape derivation and RoPE helpers work
unchanged.

``vision_config`` is parsed into :class:`MuseSparkOpenVisionConfig`, which
flattens the nested ``navit_config`` and the three fields that sit one level
above it (``patch_size``, ``patch_temporal``, ``downsample_factor``) plus
``projector_input_dim`` onto one object. The audio sub-config is still
preserved verbatim.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers.configuration_utils import PretrainedConfig

# Attention scope entries in ``global_attn_cfg``: a positive int is a sliding
# window of that many keys (including the query's own key); 0 means full causal
# attention.
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


class MuseSparkOpenVisionConfig(PretrainedConfig):
    """NaViT vision-tower config, flattened from ``vision_config``.

    The checkpoint splits the tower's knobs over two levels: most live in
    ``vision_config.navit_config``, while ``patch_size``, ``patch_temporal``,
    ``downsample_factor`` are duplicated one level up and
    ``projector_input_dim`` exists *only* one level up. :meth:`from_hf` folds
    both levels into a single object and rejects a disagreement between the
    duplicated pairs rather than silently preferring one.

    Two shapes are derived, not read: ``head_dim`` (no such key exists in the
    checkpoint) and the identity ``hidden_size == vision_output_dim //
    downsample_factor ** 2``, which the pixel-shuffle downsample makes
    mandatory -- the tower runs at ``hidden_size`` and emits
    ``vision_output_dim`` only because ``downsample_factor ** 2`` neighbouring
    patches are concatenated along the feature axis.
    """

    model_type = "musespark_open_navit"

    def __init__(
        self,
        hidden_size: int = 1536,
        num_hidden_layers: int = 50,
        num_attention_heads: int = 16,
        intermediate_size: int = 8960,
        num_channels: int = 3,
        norm_eps: float = 1e-5,
        patch_size: int = 14,
        image_size: int = 448,
        patch_temporal: int = 2,
        downsample_factor: int = 2,
        video_downsample_factor: int = 2,
        downsample_strategy: str = "ps",
        sparse_attention_factor: int = 4,
        use_rope: bool = True,
        rope_theta: float = 10000.0,
        use_cls_token: bool = False,
        grid_sample_mode: str = "bilinear",
        hidden_act: str = "gelu",
        mlp_ratio: float = 4.0,
        vision_output_dim: int = 6144,
        projector_input_dim: int = 4096,
        use_rowwise_vision_projection: bool = True,
        adapter_type: str = "simple_mlp",
        image_mean: Optional[Sequence[float]] = None,
        image_std: Optional[Sequence[float]] = None,
        min_num_image_tokens: int = 1,
        max_num_image_tokens: int = 4096,
        max_num_video_tokens: int = 4096,
        **kwargs,
    ):
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.intermediate_size = int(intermediate_size)
        self.num_channels = int(num_channels)
        self.norm_eps = float(norm_eps)
        self.patch_size = int(patch_size)
        self.image_size = int(image_size)
        self.patch_temporal = int(patch_temporal)
        self.downsample_factor = int(downsample_factor)
        self.video_downsample_factor = int(video_downsample_factor)
        self.downsample_strategy = str(downsample_strategy)
        self.sparse_attention_factor = int(sparse_attention_factor)
        self.use_rope = bool(use_rope)
        self.rope_theta = float(rope_theta)
        self.use_cls_token = bool(use_cls_token)
        self.grid_sample_mode = str(grid_sample_mode)
        self.hidden_act = str(hidden_act)
        # Kept because the checkpoint carries it, but deliberately NOT used to
        # size the MLP: 4.0 * 1536 = 6144 != intermediate_size 8960.
        self.mlp_ratio = float(mlp_ratio)
        self.vision_output_dim = int(vision_output_dim)
        self.projector_input_dim = int(projector_input_dim)
        self.use_rowwise_vision_projection = bool(use_rowwise_vision_projection)
        self.adapter_type = str(adapter_type)
        self.image_mean = list(image_mean if image_mean is not None else (0.5, 0.5, 0.5))
        self.image_std = list(image_std if image_std is not None else (0.5, 0.5, 0.5))
        self.min_num_image_tokens = int(min_num_image_tokens)
        self.max_num_image_tokens = int(max_num_image_tokens)
        self.max_num_video_tokens = int(max_num_video_tokens)

        if self.num_attention_heads <= 0 or self.hidden_size % self.num_attention_heads:
            raise ValueError(
                f"MuseSparkOpen vision hidden_size {self.hidden_size} is not divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        # No head_dim key exists in the checkpoint; 1536 / 16 = 96.
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim % 4:
            raise ValueError(
                f"MuseSparkOpen vision head_dim {self.head_dim} must be divisible by 4: the "
                "axial 2D RoPE spends head_dim // 4 base frequencies per axis"
            )

        if self.downsample_strategy != "ps":
            raise ValueError(
                "MuseSparkOpen supports downsample_strategy='ps' (pixel shuffle), got "
                f"{self.downsample_strategy!r}"
            )
        # Exact, not floor division: `vision_output_dim // downsample_factor ** 2`
        # would accept the three values just above a multiple of four.
        if self.hidden_size * self.downsample_factor**2 != self.vision_output_dim:
            raise ValueError(
                f"MuseSparkOpen vision hidden_size {self.hidden_size} * downsample_factor**2 "
                f"{self.downsample_factor**2} != vision_output_dim {self.vision_output_dim}; "
                "the pixel-shuffle downsample is the only reason those two differ"
            )
        if self.use_cls_token:
            raise ValueError("MuseSparkOpen vision tower has no CLS token (use_cls_token=True)")
        if self.grid_sample_mode != "bilinear":
            raise ValueError(
                "MuseSparkOpen positional-embedding interpolation is bilinear, got "
                f"{self.grid_sample_mode!r}"
            )
        if self.hidden_act != "gelu":
            raise ValueError(
                f"MuseSparkOpen vision tower uses exact-erf GELU, got {self.hidden_act!r}"
            )
        if self.image_size % self.patch_size:
            raise ValueError(
                f"MuseSparkOpen vision image_size {self.image_size} is not a whole number of "
                f"patch_size {self.patch_size} patches"
            )
        # The block-local attention window is one training-resolution image:
        # 448 / 14 = 32 patches per side. Also the side of the learned
        # positional-embedding grid that grid_sample interpolates from.
        self.pos_emb_grid_size = self.image_size // self.patch_size

        if kwargs.get("torch_dtype", None) is None:
            kwargs["torch_dtype"] = torch.bfloat16
        super().__init__(**kwargs)

    @property
    def full_attention_layers(self) -> List[int]:
        """Layer indices that run full per-image attention.

        Every other layer is restricted to ``pos_emb_grid_size`` -square patch
        windows. With 50 layers and ``sparse_attention_factor`` 4 this is
        ``[3, 7, ..., 47, 49]`` -- 13 of 50.
        """
        return [
            i
            for i in range(self.num_hidden_layers)
            if layer_uses_full_vision_attention(
                i, self.num_hidden_layers, self.sparse_attention_factor
            )
        ]

    @classmethod
    def from_hf(cls, vision_config: Dict[str, Any]) -> "MuseSparkOpenVisionConfig":
        """Flatten the checkpoint's two-level ``vision_config`` into one object."""
        outer = dict(vision_config)
        navit = dict(outer.pop("navit_config", None) or {})
        # ``llm_hidden_size`` is null in the checkpoint and unread by both
        # reference implementations; drop it rather than carry a null through.
        navit.pop("llm_hidden_size", None)
        for shared in ("patch_size", "patch_temporal", "downsample_factor"):
            if shared in outer and shared in navit and outer[shared] != navit[shared]:
                raise ValueError(
                    f"MuseSparkOpen vision_config.{shared}={outer[shared]!r} disagrees with "
                    f"vision_config.navit_config.{shared}={navit[shared]!r}"
                )
        merged = {**outer, **navit}
        return cls(**merged)


def layer_uses_full_vision_attention(
    layer_idx: int, num_hidden_layers: int, sparse_attention_factor: int
) -> bool:
    """Whether vision layer ``layer_idx`` attends over the whole image.

    Mirrors the reference schedule exactly: a layer is full when sparsity is
    disabled, when it is the last layer, or when ``(i + 1)`` is a multiple of
    ``sparse_attention_factor``. Note the off-by-one -- it is ``i + 1``, not
    ``i``, so the first full layer is index 3, not 0.
    """
    if sparse_attention_factor <= 1:
        return True
    if layer_idx == num_hidden_layers - 1:
        return True
    return (layer_idx + 1) % sparse_attention_factor == 0


def nope_offset(num_hidden_layers: int, every_n_layers_nope: int) -> int:
    """Offset that aligns the window schedule with the training-side ordering.

    Mirrors ``count_backward_layer_id`` in the source model: the layer index is
    shifted by ``every_n_layers_nope - num_hidden_layers % every_n_layers_nope``
    before indexing ``global_attn_cfg``.
    """
    return every_n_layers_nope - num_hidden_layers % every_n_layers_nope


def layer_uses_rope(layer_idx: int, num_hidden_layers: int, every_n_layers_nope: int) -> bool:
    """Whether ``layer_idx`` applies RoPE (NoPE layers return False)."""
    return (num_hidden_layers - layer_idx - 1) % every_n_layers_nope != 0


def layer_window(
    layer_idx: int, num_hidden_layers: int, every_n_layers_nope: int, global_attn_cfg: List[int]
) -> Optional[int]:
    """Sliding window for ``layer_idx``; None for full causal attention."""
    entry = global_attn_cfg[
        (layer_idx + nope_offset(num_hidden_layers, every_n_layers_nope)) % len(global_attn_cfg)
    ]
    if isinstance(entry, int) and entry > 0:
        return entry
    return None


class MuseSparkOpenConfig(PretrainedConfig):
    model_type = "musespark_open_omni"

    def __init__(
        self,
        vocab_size: int = 202048,
        hidden_size: int = 8192,
        num_hidden_layers: int = 62,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 16,
        head_dim: int = 64,
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-5,
        rms_post_norm_eps: float = 1e-8,
        rope_theta: float = 500000.0,
        every_n_layers_nope: int = 4,
        global_attn_cfg: Optional[List[int]] = None,
        moe: Optional[Dict[str, Any]] = None,
        metap: Optional[Dict[str, Any]] = None,
        numerics_args: Optional[Dict[str, Any]] = None,
        non_linearity: str = "swiglu",
        use_qk_norm: bool = True,
        use_rope: bool = True,
        use_residual_gate: bool = True,
        use_attn_o_gate: bool = True,
        use_attn_o_norm: bool = True,
        use_sqrt_attn_scaling: bool = False,
        use_ffn_learnable_scales: bool = True,
        norm_zero_centered_gamma: bool = True,
        normalize_tok_embeddings: bool = True,
        post_norm_affine: bool = False,
        post_norm_gain_center_type: str = "one",
        output_norm_gain_center_type: str = "zero",
        output_soft_cap_temp: float = 20.0,
        residual_dtype: str = "fp32",
        rope_use_fp32_in_outer_product: bool = True,
        tie_word_embeddings: bool = False,
        vision_config: Optional[Union[Dict[str, Any], MuseSparkOpenVisionConfig]] = None,
        image_token_id: int = 200090,
        video_token_id: int = 200091,
        audio_token_id: int = 200038,
        image_start_token_id: int = 200080,
        image_end_token_id: int = 200081,
        patch_token_id: int = 200092,
        vid_start_token_id: int = 200082,
        vid_end_token_id: int = 200083,
        vid_frame_separator_token_id: int = 200087,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rms_post_norm_eps = rms_post_norm_eps
        self.rope_theta = rope_theta
        self.every_n_layers_nope = every_n_layers_nope
        self.global_attn_cfg = list(global_attn_cfg or [2048, 2048, 2048, 0])
        self.moe = dict(moe or {})
        self.metap = dict(metap or {})
        self.numerics_args = dict(numerics_args or {})
        # The expert and dense FFNs are gated SiLU. Pinned rather than left to
        # `**kwargs`, so a checkpoint variant with a different activation fails
        # here instead of silently running SwiGLU against non-SwiGLU weights.
        if non_linearity != "swiglu":
            raise ValueError(f"MuseSparkOpen supports non_linearity='swiglu', got {non_linearity!r}")
        self.non_linearity = non_linearity
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.use_residual_gate = use_residual_gate
        self.use_attn_o_gate = use_attn_o_gate
        self.use_attn_o_norm = use_attn_o_norm
        self.use_sqrt_attn_scaling = use_sqrt_attn_scaling
        self.use_ffn_learnable_scales = use_ffn_learnable_scales
        self.norm_zero_centered_gamma = norm_zero_centered_gamma
        self.normalize_tok_embeddings = normalize_tok_embeddings
        self.post_norm_affine = post_norm_affine
        self.post_norm_gain_center_type = post_norm_gain_center_type
        self.output_norm_gain_center_type = output_norm_gain_center_type
        self.output_soft_cap_temp = output_soft_cap_temp
        self.residual_dtype = residual_dtype
        self.rope_use_fp32_in_outer_product = rope_use_fp32_in_outer_product

        # ---- multimodal -----------------------------------------------------
        # Only the first three ids exist in config.json; the six delimiter ids
        # are fixed additions in the checkpoint's tokenizer, restated here so
        # the input processor reads one object instead of re-deriving them.
        # `<|image|>` is a content marker in the chat template and is never
        # emitted into the prompt: an image expands to `<|image_start|>` +
        # `<|patch|>` * N + `<|image_end|>`.
        self.image_token_id = int(image_token_id)
        self.video_token_id = int(video_token_id)
        self.audio_token_id = int(audio_token_id)
        self.image_start_token_id = int(image_start_token_id)
        self.image_end_token_id = int(image_end_token_id)
        self.patch_token_id = int(patch_token_id)
        self.vid_start_token_id = int(vid_start_token_id)
        self.vid_end_token_id = int(vid_end_token_id)
        self.vid_frame_separator_token_id = int(vid_frame_separator_token_id)
        if isinstance(vision_config, MuseSparkOpenVisionConfig):
            self.vision_config = vision_config
        elif vision_config is None:
            self.vision_config = None
        else:
            self.vision_config = MuseSparkOpenVisionConfig.from_hf(vision_config)

        # ---- MoE, flattened -------------------------------------------------
        self.num_experts = int(self.moe.get("num_experts", 256))
        self.moe_topk = int(self.moe.get("topk", 8))
        # Aliases the generic MoE shape derivation looks for.
        self.num_local_experts = self.num_experts
        self.num_experts_per_tok = self.moe_topk
        self.compression_factor = int(self.moe.get("compression_factor", 2))
        # Expert-space width: the pooler compresses the residual width before
        # routing, so experts run at hidden_size / compression_factor.
        self.expert_hidden_size = self.hidden_size // self.compression_factor
        # `override_routed_hidden_dim` is the expert FFN intermediate width.
        # Exposed as `intermediate_size` so derive_moe_layer_shapes and other
        # generic helpers read the right value.
        self.intermediate_size = int(
            self.moe.get("override_routed_hidden_dim") or self.expert_hidden_size
        )
        self.router_score_division_eps = float(self.moe.get("router_score_division_eps", 1e-15))
        self.route_scale = float(self.moe.get("route_scale", 1.0))
        self.router_in_fp32 = bool(self.moe.get("router_in_fp32", True))
        self.use_pre_expert_norm = bool(self.moe.get("use_pre_expert_norm", True))
        self.use_post_expert_norm = bool(self.moe.get("use_post_expert_norm", True))

        # ---- metap multipliers ---------------------------------------------
        base_width = float(self.metap.get("base_width", 256.0))
        self.m_emb = float(self.metap.get("m_emb", 1.0))
        # (hidden / base_width) ** -0.5, applied to router logits before the
        # sigmoid and to the LM head output before the soft cap.
        self.width_mult = self.hidden_size / base_width
        self.metap_multiplier = self.width_mult**-0.5

        # ---- attention schedule, flattened ----------------------------------
        windows = [
            layer_window(i, self.num_hidden_layers, self.every_n_layers_nope, self.global_attn_cfg)
            for i in range(self.num_hidden_layers)
        ]
        distinct = {w for w in windows if w is not None}
        if len(distinct) > 1:
            raise ValueError(
                "MuseSparkOpen expects one sliding-window size across local "
                f"layers, got {sorted(distinct)} from global_attn_cfg "
                f"{self.global_attn_cfg}"
            )
        self.sliding_window = distinct.pop() if distinct else None
        self.layer_types = [SLIDING_ATTENTION if w is not None else FULL_ATTENTION for w in windows]
        self.use_rope_per_layer = [
            self.use_rope and layer_uses_rope(i, self.num_hidden_layers, self.every_n_layers_nope)
            for i in range(self.num_hidden_layers)
        ]
        # Every NoPE layer must be a full-attention layer and vice versa: the
        # source derives both from the same backward-counted layer id, and a
        # split between them would silently change which keys a layer sees.
        for i, (uses_rope, layer_type) in enumerate(zip(self.use_rope_per_layer, self.layer_types)):
            if uses_rope != (layer_type == SLIDING_ATTENTION):
                raise ValueError(
                    f"MuseSparkOpen layer {i}: RoPE={uses_rope} disagrees with "
                    f"attention scope {layer_type}"
                )

        # Query pre-scale applied after QK norm, on top of the kernel's
        # 1/sqrt(head_dim). `use_sqrt_attn_scaling=False` selects the constant
        # form; the sqrt form is not present in this checkpoint.
        if self.use_sqrt_attn_scaling:
            raise NotImplementedError("MuseSparkOpen use_sqrt_attn_scaling=True is not supported")
        self.scale_query_by = 43.7840518911 / (self.head_dim**0.5)

        if kwargs.get("torch_dtype", None) is None:
            kwargs["torch_dtype"] = torch.bfloat16

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def unquantized_layers(self, exclude_modules: Optional[List[str]]) -> List[int]:
        """Decoder layers whose experts are stored unquantized.

        Derived from the checkpoint's ``hf_quant_config.json``
        ``exclude_modules`` rather than hardcoded, so a re-export with a
        different excluded set is picked up. Accepts both the raw
        bare-substring form the producer wrote and the anchored glob form
        TRT-LLM matches with.
        """
        layers = []
        for idx in range(self.num_hidden_layers):
            token = f"layers.{idx}."
            if any(token in pattern for pattern in (exclude_modules or [])):
                layers.append(idx)
        return layers
