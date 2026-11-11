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
"""Config for Muse Spark Open (``model_type: muse_spark``).

The checkpoint's ``config.json`` is a composite config: nested
``text_config``, ``vision_config`` and ``audio_config``, with the attention
schedule stated per layer (``layer_types`` + ``layer_rope_theta``). The runtime
reads :class:`MuseSparkOpenConfig` instead, a flat config that groups the
decoder knobs under ``moe`` and ``metap`` and states the attention schedule as
a periodic rule (``every_n_layers_nope`` + ``global_attn_cfg``).
:meth:`MuseSparkOpenConfig.from_open_format` translates the first into the
second and checks every field the runtime implements. The flat config also
publishes the ``layer_types`` / ``sliding_window`` pair, so the generic KV-cache
window derivation, MoE shape derivation and RoPE helpers work unchanged.

The vision tower's config is :class:`MuseSparkOpenVisionConfig`. The audio
tower is fixed, and its config is only checked.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from transformers.configuration_utils import PretrainedConfig

# Attention scope entries in ``global_attn_cfg``: a positive int is a sliding
# window of that many keys (including the query's own key); 0 means full causal
# attention.
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


class MuseSparkOpenVisionConfig(PretrainedConfig):
    """NaViT vision-tower config.

    :meth:`from_hf` builds it from a two-level dict: most knobs live in
    ``navit_config``, while ``patch_size``, ``patch_temporal`` and
    ``downsample_factor`` are duplicated one level up and
    ``projector_input_dim`` exists *only* one level up. It folds both levels
    into a single object and rejects a disagreement between the duplicated
    pairs rather than silently preferring one.

    Two shapes are derived, not read: ``head_dim`` (no such key exists in the
    checkpoint) and the identity ``hidden_size == vision_output_dim //
    downsample_factor ** 2``, which the pixel-shuffle downsample makes
    mandatory -- the tower runs at ``hidden_size`` and emits
    ``vision_output_dim`` only because ``downsample_factor ** 2`` neighbouring
    patches are concatenated along the feature axis.
    """

    model_type = "muse_spark_vision"

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
        # Kept for completeness, but deliberately NOT used to size the MLP:
        # 4.0 * 1536 = 6144 != intermediate_size 8960.
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
        """Flatten a two-level ``vision_config`` dict into one object."""
        outer = dict(vision_config)
        navit = dict(outer.pop("navit_config", None) or {})
        # ``llm_hidden_size`` is unused; drop it rather than carry a null through.
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

    A layer is full when sparsity is disabled, when it is the last layer, or
    when ``(i + 1)`` is a multiple of
    ``sparse_attention_factor``. Note the off-by-one -- it is ``i + 1``, not
    ``i``, so the first full layer is index 3, not 0.
    """
    if sparse_attention_factor <= 1:
        return True
    if layer_idx == num_hidden_layers - 1:
        return True
    return (layer_idx + 1) % sparse_attention_factor == 0


def nope_offset(num_hidden_layers: int, every_n_layers_nope: int) -> int:
    """Offset that aligns the window schedule with layers counted from the end.

    The schedule is anchored on the last layer, so the layer index is shifted
    by ``every_n_layers_nope - num_hidden_layers % every_n_layers_nope`` before
    indexing ``global_attn_cfg``.
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


def _require(mapping: Dict[str, Any], key: str, expected: Any, why: str) -> None:
    """Reject an open-format field whose value the runtime cannot honor.

    A missing key passes: every checked field has exactly one value the runtime
    implements, so absence and agreement are the same statement. A present but
    different value is a different model and must fail here, not load quietly.
    """
    actual = mapping.get(key, expected)
    if actual != expected:
        raise ValueError(
            f"Muse Spark Open format: {key}={actual!r} but the runtime implements "
            f"{expected!r} ({why})"
        )


def _derive_nope_schedule(
    layer_types: Sequence[str],
    layer_rope_theta: Sequence[float],
    sliding_window: Optional[int],
    rope_theta: float,
) -> Tuple[int, List[int]]:
    """Invert per-layer ``layer_types``/``layer_rope_theta`` to the flat form.

    The checkpoint states the attention schedule extensionally; the runtime
    implements it as a rule (``every_n_layers_nope`` + ``global_attn_cfg``,
    indexed through :func:`nope_offset`). This inverts the extensional form:
    NoPE layers are where ``layer_rope_theta`` is 0, their spacing is
    ``every_n_layers_nope``, and each residue class of the shifted index gets
    one ``global_attn_cfg`` entry. The caller re-derives ``layer_types`` from
    the result and compares, so an inconsistent schedule cannot slip through --
    this function only has to find the parameters, not to prove them.
    """
    num_layers = len(layer_types)
    if len(layer_rope_theta) != num_layers:
        raise ValueError(
            f"Muse Spark Open format: layer_rope_theta has {len(layer_rope_theta)} entries "
            f"for {num_layers} layers"
        )
    nope_layers = [i for i, theta in enumerate(layer_rope_theta) if theta == 0]
    if not nope_layers:
        raise ValueError("Muse Spark Open format: no NoPE layers (layer_rope_theta has no 0)")
    for i, theta in enumerate(layer_rope_theta):
        if theta != 0 and theta != rope_theta:
            raise ValueError(
                f"Muse Spark Open format: layer {i} rope theta {theta} != the model-wide "
                f"rope_theta {rope_theta}; per-layer theta values have no home in the "
                "flat config"
            )
    # NoPE layers sit every ``every_n_layers_nope`` layers counted from the
    # last layer backwards (see layer_uses_rope). Their spacing is the period.
    if len(nope_layers) < 2:
        raise ValueError(
            f"Muse Spark Open format: layer {nope_layers[0]} is the only NoPE layer; "
            "the NoPE period cannot be inferred from a single layer"
        )
    spacings = {b - a for a, b in zip(nope_layers, nope_layers[1:])}
    if len(spacings) != 1:
        raise ValueError(
            f"Muse Spark Open format: NoPE layers {nope_layers} are not equally spaced; "
            "the periodic schedule the runtime implements cannot express this"
        )
    every_n = spacings.pop()
    if nope_layers[-1] != num_layers - 1:
        raise ValueError(
            f"Muse Spark Open format: the last layer ({num_layers - 1}) is not NoPE; "
            "the runtime's backward-counted schedule always ends on one"
        )

    offset = nope_offset(num_layers, every_n)
    global_attn_cfg: List[Optional[int]] = [None] * every_n
    for i, layer_type in enumerate(layer_types):
        if layer_type == FULL_ATTENTION:
            entry = 0
        elif layer_type == SLIDING_ATTENTION:
            if not sliding_window:
                raise ValueError(
                    "Muse Spark Open format: sliding_attention layers but no sliding_window"
                )
            entry = int(sliding_window)
        else:
            raise ValueError(
                f"Muse Spark Open format: unknown layer_types entry {layer_type!r} at layer {i}"
            )
        slot = (i + offset) % every_n
        if global_attn_cfg[slot] is None:
            global_attn_cfg[slot] = entry
        elif global_attn_cfg[slot] != entry:
            raise ValueError(
                f"Muse Spark Open format: layers in residue class {slot} mix attention "
                "scopes; the periodic schedule the runtime implements cannot express this"
            )
    if any(entry is None for entry in global_attn_cfg):
        raise ValueError(
            "Muse Spark Open format: not every global_attn_cfg slot is populated by "
            f"{num_layers} layers with period {every_n}"
        )
    return every_n, list(global_attn_cfg)


def _vision_config_from_open_format(config_dict: Dict[str, Any]) -> MuseSparkOpenVisionConfig:
    """Translate the checkpoint's ``vision_config`` to the runtime's tower config.

    The checkpoint names ``downsample_factor`` ``merge_size`` and
    ``norm_eps`` to ``layer_norm_eps``, expresses the attention schedule as
    ``layer_types`` instead of ``sparse_attention_factor``, publishes the
    positional-embedding grid instead of ``image_size``, and hoists the two
    projector widths to the top level (``out_hidden_size``,
    ``projector_hidden_size``).
    """
    vision = dict(config_dict.get("vision_config") or {})
    if not vision:
        raise ValueError("Muse Spark Open format: config has no vision_config")

    num_layers = int(vision["num_hidden_layers"])
    layer_types = list(vision.get("layer_types") or [])
    if len(layer_types) != num_layers:
        raise ValueError(
            f"Muse Spark Open format: vision layer_types has {len(layer_types)} entries "
            f"for {num_layers} layers"
        )
    full_layers = [i for i, t in enumerate(layer_types) if t == FULL_ATTENTION]
    if not full_layers:
        raise ValueError("Muse Spark Open format: vision tower has no full-attention layer")
    # The first full layer under this schedule is index factor-1.
    factor = full_layers[0] + 1
    expected = [
        FULL_ATTENTION
        if layer_uses_full_vision_attention(i, num_layers, factor)
        else "window_attention"
        for i in range(num_layers)
    ]
    if layer_types != expected:
        raise ValueError(
            "Muse Spark Open format: vision layer_types do not follow the "
            f"sparse_attention_factor={factor} schedule the runtime implements"
        )

    rope_parameters = vision.get("rope_parameters") or {}
    _require(rope_parameters, "rope_type", "axial", "the tower's 2D RoPE form")

    pos_emb_height = int(vision.get("pos_emb_height", 32))
    pos_emb_width = int(vision.get("pos_emb_width", pos_emb_height))
    if pos_emb_height != pos_emb_width:
        raise ValueError(
            f"Muse Spark Open format: vision pos_emb grid {pos_emb_height}x{pos_emb_width} "
            "is not square; the tower interpolates from a square grid"
        )

    merge_size = int(vision.get("merge_size", 2))
    return MuseSparkOpenVisionConfig(
        hidden_size=int(vision["hidden_size"]),
        num_hidden_layers=num_layers,
        num_attention_heads=int(vision["num_attention_heads"]),
        intermediate_size=int(vision["intermediate_size"]),
        norm_eps=float(vision.get("layer_norm_eps", 1e-5)),
        patch_size=int(vision["patch_size"]),
        image_size=pos_emb_height * int(vision["patch_size"]),
        patch_temporal=int(vision.get("patch_temporal", 2)),
        downsample_factor=merge_size,
        video_downsample_factor=merge_size,
        sparse_attention_factor=factor,
        rope_theta=float(rope_parameters.get("rope_theta", 10000.0)),
        hidden_act=str(vision.get("hidden_act", "gelu")),
        vision_output_dim=int(config_dict["out_hidden_size"]),
        projector_input_dim=int(config_dict["projector_hidden_size"]),
    )


def _check_audio_config_matches_encoder(
    audio_config: Optional[Dict[str, Any]], hidden_size: int
) -> None:
    """Reject an ``audio_config`` the fixed wav_ae tower cannot realize.

    The runtime's audio encoder hardcodes its stage plan; the checkpoint
    publishes the same plan as data. There is nothing to translate -- only to
    verify, so a re-export with a different tower fails at config time instead
    of at the weight walk.
    """
    audio = dict(audio_config or {})
    if not audio:
        raise ValueError("Muse Spark Open format: config has no audio_config")
    _require(audio, "patch_sizes", [16, 6, 5, 4], "the wav_ae patchify plan")
    _require(audio, "hidden_sizes", [256, 512, 1024, 2048], "the wav_ae stage widths")
    _require(audio, "depths", [2, 2, 2, 2], "the wav_ae blocks per stage")
    _require(audio, "conv_kernel_size", 7, "the wav_ae depthwise kernel")
    _require(audio, "hidden_act", "gelu", "the wav_ae activation")
    _require(audio, "mlp_ratio", 4.0, "the wav_ae MLP width ratio")
    _require(audio, "out_hidden_size", hidden_size, "the decoder width the tower emits")


class MuseSparkOpenConfig(PretrainedConfig):
    """Flat runtime config; :meth:`from_open_format` builds it from a checkpoint."""

    model_type = "muse_spark"

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
            raise ValueError(
                f"MuseSparkOpen supports non_linearity='swiglu', got {non_linearity!r}"
            )
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
        # The delimiter ids are fixed tokens of the checkpoint's tokenizer,
        # restated here so the input processor reads one object instead of
        # re-deriving them. `<|image|>` is a content marker in the chat
        # template and is never emitted into the prompt: an image expands to
        # `<|image_start|>` + `<|patch|>` * N + `<|image_end|>`.
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
        # Every NoPE layer must be a full-attention layer and vice versa: both
        # follow from the same backward-counted layer index, and a split
        # between them would silently change which keys a layer sees.
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

    @classmethod
    def from_open_format(cls, config_dict: Dict[str, Any]) -> "MuseSparkOpenConfig":
        """Build the flat config from the checkpoint's composite config.

        The checkpoint publishes the model in HF composite form:
        ``model_type: muse_spark`` with nested ``text_config`` /
        ``vision_config`` / ``audio_config``, per-layer ``layer_types`` +
        ``layer_rope_theta`` instead of ``global_attn_cfg`` +
        ``every_n_layers_nope``, and its own names for the scalar knobs. This
        translates that layout into the flat config the runtime reads and
        cross-checks every derivable quantity, so a re-export that changes the
        model (not just the spelling) fails here instead of loading wrong.

        Deliberately dropped fields, none of which carry information the
        runtime uses:

        * ``text_config.intermediate_size`` (16384): the dense-FFN width for
          ``mlp_layer_types == "dense"`` layers, of which this model has none
          (asserted). The flat config's ``intermediate_size`` is the routed
          expert width, derived from ``moe_intermediate_size``.
        * ``transformers_version``, ``initializer_range``: bookkeeping.

        ``quantization_config`` is kept as is. ``ModelConfig.from_pretrained``
        treats ``hf_quant_config.json`` as authoritative, warns when the inline
        copy disagrees with it, and falls back to the inline copy when the file
        is absent or sets no global ``quant_algo``.
        """
        text = dict(config_dict.get("text_config") or {})
        if not text:
            raise ValueError("Muse Spark Open format: config has no text_config")
        # The class constructor defaults torch_dtype to bfloat16; the
        # checkpoint states the same at the top level. A different dtype, under
        # either key name, must not be silently overridden by that default.
        for dtype_key in ("dtype", "torch_dtype"):
            _require(config_dict, dtype_key, "bfloat16", "the export dtype the runtime assumes")

        num_hidden_layers = int(text["num_hidden_layers"])
        hidden_size = int(text["hidden_size"])
        head_dim = int(text["head_dim"])

        _require(text, "hidden_act", "silu", "the decoder FFNs are gated SiLU (swiglu)")
        _require(text, "attention_bias", False, "the runtime attention has no bias")
        _require(text, "attention_dropout", 0.0, "the runtime attention has no dropout")
        _require(text, "num_shared_experts", 0, "the MoE has no shared expert")
        if set(text.get("mlp_layer_types") or []) != {"sparse"}:
            raise ValueError(
                "Muse Spark Open format: expected every mlp_layer_types entry to be "
                f"'sparse', got {sorted(set(text.get('mlp_layer_types') or []))}; a dense "
                "FFN layer has no home in the flat config"
            )

        rope_parameters = text.get("rope_parameters") or {}
        _require(rope_parameters, "rope_type", "default", "the decoder RoPE has no scaling")
        rope_theta = float(rope_parameters.get("rope_theta", 500000.0))
        every_n_layers_nope, global_attn_cfg = _derive_nope_schedule(
            text["layer_types"],
            text["layer_rope_theta"],
            text.get("sliding_window"),
            rope_theta,
        )

        # The temperature in the gated residual's ``sigmoid(g / 0.3)`` is a
        # checkpoint property the runtime hardcodes; a checkpoint trained with
        # another value must not load silently.
        _require(text, "residual_gate_temperature", 0.3, "the gated-residual temperature")

        moe = {
            "topk": int(text["num_experts_per_tok"]),
            "num_experts": int(text["num_local_experts"]),
            "route_scale": float(text.get("routed_scaling_factor", 1.0)),
            "override_routed_hidden_dim": int(text["moe_intermediate_size"]),
        }
        moe_hidden_size = int(text["moe_hidden_size"])
        if moe_hidden_size <= 0 or hidden_size % moe_hidden_size:
            raise ValueError(
                f"Muse Spark Open format: hidden_size {hidden_size} is not a whole "
                f"multiple of moe_hidden_size {moe_hidden_size}"
            )
        moe["compression_factor"] = hidden_size // moe_hidden_size

        # metap ships as its consequence: output_multiplier == (hidden /
        # base_width) ** -0.5. Invert it and cross-check the round-trip.
        output_multiplier = float(text["output_multiplier"])
        base_width = hidden_size * output_multiplier**2
        if abs(base_width - round(base_width)) > 1e-6:
            raise ValueError(
                f"Muse Spark Open format: output_multiplier {output_multiplier} does not "
                f"invert to an integral metap base_width (got {base_width})"
            )
        metap = {"base_width": float(round(base_width)), "m_emb": 1.0}

        vision_config = _vision_config_from_open_format(config_dict)
        _check_audio_config_matches_encoder(config_dict.get("audio_config"), hidden_size)

        flat = cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=head_dim,
            max_position_embeddings=int(text["max_position_embeddings"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            rms_post_norm_eps=float(text["post_norm_eps"]),
            rope_theta=rope_theta,
            every_n_layers_nope=every_n_layers_nope,
            global_attn_cfg=global_attn_cfg,
            moe=moe,
            metap=metap,
            output_soft_cap_temp=float(text["final_logit_softcapping"]),
            tie_word_embeddings=bool(text.get("tie_word_embeddings", False)),
            vision_config=vision_config,
            bos_token_id=text.get("bos_token_id"),
            eos_token_id=text.get("eos_token_id"),
            pad_token_id=text.get("pad_token_id"),
        )
        if config_dict.get("quantization_config") is not None:
            flat.quantization_config = dict(config_dict["quantization_config"])

        # The audio tower is enabled and configured through this dict;
        # audio_config was checked against the same fixed tower above.
        flat.audio_perception_modality_args = {
            "encoder_family": "wav_ae",
            "use_patch_encoder": False,
            "input_mode": "wav",
            "next_audio_token": 201471,
            "empty_audio_token": 201472,
        }
        flat.use_audio_perception = True

        # Round-trip checks: the flat config re-derives the per-layer schedule
        # and the two scalar multipliers from what was passed above; they must
        # reproduce the checkpoint's own statements exactly.
        if flat.layer_types != list(text["layer_types"]):
            raise ValueError(
                "Muse Spark Open format: derived layer_types disagree with the "
                "checkpoint's layer_types; the attention schedule is not the "
                f"every_n_layers_nope={every_n_layers_nope} / "
                f"global_attn_cfg={global_attn_cfg} form the runtime implements"
            )
        expected_rope = [theta != 0 for theta in text["layer_rope_theta"]]
        if flat.use_rope_per_layer != expected_rope:
            raise ValueError(
                "Muse Spark Open format: derived RoPE schedule disagrees with layer_rope_theta"
            )
        qk_scale_factor = float(text["qk_scale_factor"])
        if abs(qk_scale_factor - flat.scale_query_by) > 1e-9 * abs(flat.scale_query_by):
            raise ValueError(
                f"Muse Spark Open format: qk_scale_factor {qk_scale_factor} != the "
                f"runtime's query pre-scale {flat.scale_query_by}"
            )
        if abs(output_multiplier - flat.metap_multiplier) > 1e-9 * abs(flat.metap_multiplier):
            raise ValueError(
                f"Muse Spark Open format: output_multiplier {output_multiplier} != the "
                f"derived metap multiplier {flat.metap_multiplier}"
            )

        # The checkpoint's ``image_token_id`` is the ``<|patch|>`` marker, which
        # the flat config calls ``patch_token_id`` (its ``image_token_id`` is the
        # ``<|image|>`` content marker). The ids are fixed, and the flat defaults
        # already carry them.
        _require(config_dict, "image_token_id", flat.patch_token_id, "the <|patch|> marker id")
        _require(config_dict, "video_token_id", flat.video_token_id, "the <|video|> marker id")
        _require(config_dict, "audio_token_id", flat.audio_token_id, "the <|audio|> marker id")
        return flat
