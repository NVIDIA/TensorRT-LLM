# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Multimodal entry point for the Step3p7 Flash checkpoint.

Wraps the text-only ``Step3p7ForCausalLM`` (text decoder + MTP) plus a
Perception-Encoder vision tower. When a request has no multimodal payload
this wrapper is a thin passthrough so plain text generation keeps the
original Step3p7 text path.

The vision tower mirrors the HF reference checkpoint's ``vision_encoder.py``:
``StepRoboticsVisionEncoder`` is a Perception-Encoder-style ViT (patch
embedding via Conv2d, 47 transformer blocks with 2D RoPE, two trailing
Conv2d downsamplers).  The matching projector ``vit_large_projector`` is a
single bf16 Linear from ``4 * width`` to ``text_config.hidden_size``.

The vision tower is intentionally kept in raw torch (SDPA for non-causal
attention) instead of TensorRT-LLM's ``Attention`` module.  ``Attention``
specialises for causal/MLA text decoders; a faithful port of the HF code
keeps weight names trivially compatible (``vision_model.transformer.resblocks.<i>
.attn.{in_proj_weight,in_proj_bias,out_proj.{weight,bias}}``) and avoids
plumbing the vision attention through the text attention metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..._utils import nvtx_range
from ...inputs import (
    BaseMultimodalInputProcessor,
    ContentFormat,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    TextPrompt,
    register_input_processor,
)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.layer_norm import LayerNorm
from ..speculative import SpecMetadata
from .modeling_multimodal_utils import (
    _is_mm_disagg,
    find_input_mm_embeds,
    fuse_input_embeds,
    get_multimodal_embeddings,
)
from .modeling_step3p7 import (
    Step3p7ForCausalLM,
    _get_text_config,
    _mirror_step3p7_text_aliases,
    _normalize_torch_dtype,
)
from .modeling_utils import register_auto_model, register_vision_encoder

# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dim halves -- 2D RoPE helper used by the vision tower."""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], -1)


def _apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos()) + (_rotate_half(t_rot) * freqs.sin())
    return torch.cat((t_rot, t_pass), dim=-1).to(dtype)


class Step3VisionRope2D(nn.Module):
    """Cached 2D rotary positional embedding for the vision tower."""

    def __init__(
        self,
        dim: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool = False,
        theta: float = 10000.0,
        theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_grid_height = max_grid_height
        self.max_grid_width = max_grid_width
        self.use_cls_token = use_cls_token
        self.theta = theta * theta_rescale_factor ** (dim / (dim - 2))
        self.register_buffer("freqs_cache", self._compute_2d_freqs(), persistent=False)

    def _compute_inv_freq(self, base: float, dim: int) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    def _compute_freqs(self, t: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        freqs = torch.einsum("..., f -> ... f", t.type(inv_freq.dtype), inv_freq)
        return freqs.repeat_interleave(2, dim=-1)

    def _compute_2d_freqs(self) -> torch.Tensor:
        grid_h = torch.arange(self.max_grid_height, dtype=torch.float)
        grid_w = torch.arange(self.max_grid_width, dtype=torch.float)
        if self.use_cls_token:
            grid_h = grid_h + 1
            grid_w = grid_w + 1
        inv_freq = self._compute_inv_freq(self.theta, self.dim // 2)
        freqs_h = self._compute_freqs(grid_h, inv_freq)[:, None].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs_w = self._compute_freqs(grid_w, inv_freq)[None, :].expand(
            self.max_grid_height, self.max_grid_width, -1
        )
        freqs = torch.cat([freqs_w, freqs_h], dim=-1).reshape(
            self.max_grid_height * self.max_grid_width, -1
        )
        if self.use_cls_token:
            freqs = torch.cat([torch.zeros(1, freqs.shape[-1]), freqs], dim=0)
        return freqs[None, None, ...]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=q.device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=q.device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            if self.use_cls_token:
                positions = torch.cat(
                    [
                        torch.zeros(1, device=q.device, dtype=torch.long),
                        positions + 1,
                    ],
                    dim=0,
                )
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        return _apply_rotary_emb(freqs, q), _apply_rotary_emb(freqs, k)


class Step3VisionLayerScale(nn.Module):
    """Per-channel residual scaling used when ``ls_init_value`` is set."""

    def __init__(self, dim: int, init_value: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.gamma


class Step3VisionMLP(nn.Module):
    """``c_fc -> act -> c_proj`` FFN matching the HF weight names."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.c_fc = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.act_fn = ACT2FN[hidden_act]
        self.c_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act_fn(self.c_fc(hidden_states)))


class Step3VisionAttention(nn.Module):
    """Vision MHA with 2D RoPE.

    HF stores the fused QKV projection as ``in_proj_weight``/``in_proj_bias``
    (a single ``3*H x H`` matrix); we keep the same parameter names so the
    checkpoint loads without remapping.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool,
        use_rope2d: bool,
        rope_theta: float = 10000.0,
        rope_theta_rescale_factor: float = 1.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5

        # HF parameter names: in_proj_weight (3H, H), in_proj_bias (3H,), out_proj.
        self.in_proj_weight = nn.Parameter(torch.zeros(hidden_size * 3, hidden_size))
        self.in_proj_bias = nn.Parameter(torch.zeros(hidden_size * 3))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

        self.rope: Optional[Step3VisionRope2D] = None
        if use_rope2d:
            self.rope = Step3VisionRope2D(
                dim=self.head_dim,
                max_grid_height=max_grid_height,
                max_grid_width=max_grid_width,
                use_cls_token=use_cls_token,
                theta=rope_theta,
                theta_rescale_factor=rope_theta_rescale_factor,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        # TODO: port the vision attention/projector to TRT-LLM modules
        bsz, seq_len, _ = hidden_states.shape
        qkv = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q, k, grid_hw=grid_hw)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.scale)
        attn_output = attn_output.transpose(1, 2).reshape(
            bsz, seq_len, self.num_heads * self.head_dim
        )
        return self.out_proj(attn_output)


class Step3VisionBlock(nn.Module):
    """Single vision transformer block (Pre-LN + LayerScale)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: Optional[float],
        max_grid_height: int,
        max_grid_width: int,
        use_cls_token: bool,
        use_rope2d: bool,
        rope_theta: float,
        rope_theta_rescale_factor: float,
    ):
        super().__init__()
        self.attn = Step3VisionAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_grid_height=max_grid_height,
            max_grid_width=max_grid_width,
            use_cls_token=use_cls_token,
            use_rope2d=use_rope2d,
            rope_theta=rope_theta,
            rope_theta_rescale_factor=rope_theta_rescale_factor,
        )
        self.ln_1 = LayerNorm(hidden_size=hidden_size, eps=layer_norm_eps)
        self.ln_2 = LayerNorm(hidden_size=hidden_size, eps=layer_norm_eps)
        self.mlp = Step3VisionMLP(hidden_size, int(hidden_size * mlp_ratio), hidden_act)
        ls = ls_init_value if ls_init_value is not None else 1.0
        self.ls_1 = Step3VisionLayerScale(hidden_size, ls)
        self.ls_2 = Step3VisionLayerScale(hidden_size, ls)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, grid_hw=grid_hw)
        hidden_states = residual + self.ls_1(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.ls_2(hidden_states)
        return hidden_states


class Step3VisionTransformer(nn.Module):
    def __init__(self, depth: int, **block_kwargs):
        super().__init__()
        self.resblocks = nn.ModuleList([Step3VisionBlock(**block_kwargs) for _ in range(depth)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> torch.Tensor:
        for block in self.resblocks:
            hidden_states = block(hidden_states, grid_hw=grid_hw)
        return hidden_states


class Step3p7VisionEncoder(nn.Module):
    """Perception-Encoder vision tower (``vision_model.*`` checkpoint subtree).

    HF carries two trailing Conv2d downsamplers inside the same module
    (``vit_downsampler1``, ``vit_downsampler2``) but invokes them externally in
    ``Step3p7Model._process_image_features``.  We keep the same parameter
    layout so ``vision_model.vit_downsampler{1,2}.*`` weights load directly,
    but call the downsamplers from ``forward`` here so a single call returns
    the post-downsample feature map ready for the projector.
    """

    def __init__(self, vision_config: PretrainedConfig, dtype: torch.dtype):
        super().__init__()
        self.config = vision_config

        self.hidden_size = int(vision_config.width)
        self.num_heads = int(vision_config.heads)
        self.num_hidden_layers = int(vision_config.layers)
        self.patch_size = int(vision_config.patch_size)
        self.image_size = int(vision_config.image_size)
        self.layer_norm_eps = float(getattr(vision_config, "layer_norm_eps", 1e-5))
        self.hidden_act = getattr(vision_config, "hidden_act", "quick_gelu")
        self.mlp_ratio = float(getattr(vision_config, "mlp_ratio", 8960.0 / 1536.0))
        self.ls_init_value = getattr(vision_config, "ls_init_value", None)
        self.use_cls_token = bool(getattr(vision_config, "use_cls_token", False))
        self.use_rope2d = bool(getattr(vision_config, "use_rope2d", True))
        self.use_abs_posemb = bool(getattr(vision_config, "use_abs_posemb", True))
        self.use_ln_pre = bool(getattr(vision_config, "use_ln_pre", True))
        self.use_ln_post = bool(getattr(vision_config, "use_ln_post", False))

        self.conv1 = nn.Conv2d(
            in_channels=int(getattr(vision_config, "num_channels", 3)),
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.ln_pre = (
            LayerNorm(hidden_size=self.hidden_size, eps=self.layer_norm_eps)
            if self.use_ln_pre
            else nn.Identity()
        )
        self.ln_post = (
            LayerNorm(hidden_size=self.hidden_size, eps=self.layer_norm_eps)
            if self.use_ln_post
            else nn.Identity()
        )

        grid_size = self.image_size // self.patch_size
        self.base_grid = (grid_size, grid_size)

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(
                torch.randn(self.hidden_size) * (self.hidden_size**-0.5)
            )
        else:
            self.class_embedding = None

        if self.use_abs_posemb:
            self.posemb_grid_size = grid_size
            self.positional_embedding = nn.Parameter(
                (self.hidden_size**-0.5)
                * torch.randn(
                    int(self.use_cls_token) + self.posemb_grid_size**2,
                    self.hidden_size,
                )
            )
        else:
            self.posemb_grid_size = None

        self.transformer = Step3VisionTransformer(
            depth=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            ls_init_value=self.ls_init_value,
            max_grid_height=grid_size,
            max_grid_width=grid_size,
            use_cls_token=self.use_cls_token,
            use_rope2d=self.use_rope2d,
            rope_theta=float(getattr(vision_config, "rope_theta", 10000.0)),
            rope_theta_rescale_factor=float(
                getattr(vision_config, "rope_theta_rescale_factor", 1.0)
            ),
        )

        # Two trailing Conv2d downsamplers (stride 2 each, channel x2 each).
        self.vit_downsampler1 = nn.Conv2d(
            self.hidden_size,
            self.hidden_size * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.vit_downsampler2 = nn.Conv2d(
            self.hidden_size * 2,
            self.hidden_size * 4,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # Cast the whole tower to the model dtype (bf16 in practice).  The
        # PerceptionEncoder portion of the checkpoint is bf16 on disk even in
        # the FP8 text checkpoint, so a single ``to(dtype)`` is sufficient.
        self.to(dtype)
        self._dtype = dtype

    def _sample_abs_posemb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]
        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_token_embed, pos_embed = pos_embed[:1], pos_embed[1:]
        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pos_embed = F.interpolate(
            pos_embed,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)
        if self.use_cls_token:
            pos_embed = torch.cat([cls_token_embed, pos_embed], dim=0)
        return pos_embed[None, ...]

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = pixel_values.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        hidden = self.conv1(pixel_values)  # (B, D, Gh, Gw)
        hidden = hidden.flatten(2).transpose(1, 2)  # (B, Gh*Gw, D)
        if self.use_cls_token:
            cls_token = self.class_embedding.view(1, 1, -1).expand(bsz, -1, -1)
            hidden = torch.cat([cls_token, hidden], dim=1)
        if self.use_abs_posemb:
            hidden = hidden + self._sample_abs_posemb(grid_h, grid_w).to(hidden.dtype)
        hidden = self.ln_pre(hidden)
        hidden = self.transformer(hidden, grid_hw=(grid_h, grid_w))
        if self.use_ln_post:
            hidden = self.ln_post(hidden)
        if self.use_cls_token:
            hidden = hidden[:, 1:, :]
        return hidden

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision transformer + the two trailing downsamplers.

        Returns features shaped ``(B, (Gh//4) * (Gw//4), 4 * width)`` ready
        for the linear projector to map into the text hidden size.
        """
        pixel_values = pixel_values.to(dtype=self._dtype, device=self.conv1.weight.device)
        x = self.forward_features(pixel_values)  # (B, P, D)
        B, P, D = x.shape
        T = int(P**0.5)
        x = x.transpose(1, 2).contiguous().view(B, D, T, T)
        x = self.vit_downsampler1(x)
        x = self.vit_downsampler2(x)
        Bd, Cd, Td, _ = x.shape
        return x.view(Bd, Cd, Td * Td).transpose(1, 2)  # (B, T'*T', 4D)


class Step3p7VisionTower(nn.Module):
    """Vision encoder + projector exposed to TRT-LLM as the ``mm_encoder``.

    Encapsulates ``vision_model`` (``Step3p7VisionEncoder``) and the
    ``vit_large_projector`` Linear from the HF checkpoint, and provides the
    ``forward(multimodal_params)`` signature used by
    ``get_multimodal_embeddings``.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        pretrained_config = model_config.pretrained_config
        vision_config = getattr(pretrained_config, "vision_config", None)
        if vision_config is None:
            raise ValueError(
                "Step3p7VisionTower requires `vision_config` on the pretrained config; "
                "the checkpoint is missing the vision tower entry."
            )
        # Ensure vision_config carries a torch.dtype (not the raw JSON string).
        _normalize_torch_dtype(vision_config)
        # Honour the outer model dtype when set (typical bf16); fall back to
        # the vision sub-config's stored dtype.
        outer_dtype = getattr(pretrained_config, "torch_dtype", None)
        if not isinstance(outer_dtype, torch.dtype):
            outer_dtype = getattr(vision_config, "torch_dtype", None)
            if not isinstance(outer_dtype, torch.dtype):
                outer_dtype = torch.bfloat16
        self._dtype = outer_dtype

        text_config = _get_text_config(model_config)
        self.image_token_id = int(getattr(pretrained_config, "image_token_id", 128001))

        self.vision_model = Step3p7VisionEncoder(vision_config, dtype=outer_dtype)

        proj_in = 4 * int(vision_config.width)
        proj_out = int(text_config.hidden_size)
        proj_bias = bool(getattr(pretrained_config, "projector_bias", False))
        # Single GPU rank for the bring-up; keep the projector as a plain
        # bf16 Linear so weights from ``vit_large_projector.weight`` load
        # directly without an extra remapping.
        self.vit_large_projector = nn.Linear(proj_in, proj_out, bias=proj_bias).to(outer_dtype)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        """Consume ``vision_model.*`` and ``vit_large_projector.*`` keys.

        Both checkpoint layouts are supported: the FP8/BF16 exports key the
        vision subtree directly (``vision_model.*`` / ``vit_large_projector.*``),
        while the NVFP4 export nests everything under ``model.`` alongside the
        ``model.language_model.*`` text decoder (``model.vision_model.*`` /
        ``model.vit_large_projector.*``). An optional leading ``model.`` is
        stripped before matching so both load identically.

        Returns silently if neither subtree is present (e.g. text-only
        checkpoint slice); the caller will surface a missing-weights error
        downstream via the model engine if that is wrong for the scenario.
        """
        vision_state: Dict[str, torch.Tensor] = {}
        projector_state: Dict[str, torch.Tensor] = {}
        for key in list(weights.keys()):
            # Normalize the NVFP4 ``model.`` wrapper prefix to the bare layout.
            norm_key = key[len("model.") :] if key.startswith("model.") else key
            if norm_key.startswith("vision_model."):
                sub = norm_key[len("vision_model.") :]
                vision_state[sub] = weights[key]
            elif norm_key.startswith("vit_large_projector."):
                sub = norm_key[len("vit_large_projector.") :]
                projector_state[sub] = weights[key]

        if vision_state:
            self.vision_model.load_state_dict(vision_state, strict=True)
        if projector_state:
            self.vit_large_projector.load_state_dict(projector_state, strict=True)

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project post-downsample vision features to the text hidden size."""
        return self.vit_large_projector(image_features.to(self._dtype))

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.vision_model(pixel_values)
        return self._process_image_features(feats)

    @nvtx_range("Step3p7VisionTower forward()")
    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        """Encode all images (and their patches) in the batch.

        For each request we mirror the HF reference's per-image layout:
        ``[patches_for_image_i ... | full_image_i]`` flattened into a single
        ``(num_tokens, hidden)`` tensor.  The caller (``fuse_input_embeds``)
        concatenates per-request embeddings and writes them into the
        positions where the placeholder token id appears in ``input_ids``.

        Token ordering must therefore match the input processor's text
        placeholder expansion, which follows HF
        ``Step3VLProcessor._get_image_repl_features``: patch features come
        first (``num_patches`` blocks of ``num_patch_feature_size`` tokens),
        then the full image feature block (``num_image_feature_size`` tokens).
        """
        per_request_embeds: List[torch.Tensor] = []
        for mm in multimodal_params:
            # TODO: batch full images across all requests, batch patch images across all
            # requests where shapes match, and then split/reassemble according to num_patches
            image_data = mm.multimodal_data.get("image") if mm.multimodal_data else None
            if image_data is None:
                continue
            pixel_values = image_data.get("pixel_values")
            patch_pixel_values = image_data.get("patch_pixel_values")
            num_patches_list = image_data.get("num_patches")
            if pixel_values is None:
                continue

            pixel_values = pixel_values.to(self.vision_model.conv1.weight.device)
            if pixel_values.dim() >= 5:
                pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:])
            elif pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            image_feats = self._encode(pixel_values)  # (N, P_img, H)

            patch_feats = None
            if patch_pixel_values is not None and patch_pixel_values.numel() > 0:
                patch_pixel_values = patch_pixel_values.to(self.vision_model.conv1.weight.device)
                if patch_pixel_values.dim() >= 5:
                    patch_pixel_values = patch_pixel_values.view(-1, *patch_pixel_values.shape[-3:])
                elif patch_pixel_values.dim() == 3:
                    patch_pixel_values = patch_pixel_values.unsqueeze(0)
                if patch_pixel_values.shape[0] > 0:
                    patch_feats = self._encode(patch_pixel_values)  # (M, P_patch, H)

            # Build the per-request flat embedding stream:
            # patches (if any, in order) then full image, repeated per image.
            if num_patches_list is None:
                num_images = image_feats.shape[0]
                num_patches_list = [0] * num_images
            elif isinstance(num_patches_list, torch.Tensor):
                num_patches_list = num_patches_list.flatten().tolist()
            else:
                num_patches_list = list(num_patches_list)

            cur_patch_idx = 0
            flat_blocks: List[torch.Tensor] = []
            for img_idx, n_p in enumerate(num_patches_list):
                if n_p > 0 and patch_feats is not None:
                    blk = patch_feats[cur_patch_idx : cur_patch_idx + n_p]
                    flat_blocks.append(blk.reshape(-1, blk.shape[-1]))
                    cur_patch_idx += n_p
                flat_blocks.append(image_feats[img_idx].reshape(-1, image_feats.shape[-1]))

            if flat_blocks:
                per_request_embeds.append(torch.cat(flat_blocks, dim=0))

        if not per_request_embeds:
            return []
        return [torch.cat(per_request_embeds, dim=0)]


# ---------------------------------------------------------------------------
# Multimodal input processor
# ---------------------------------------------------------------------------


class Step3p7VLInputProcessor(BaseMultimodalInputProcessor):
    """Input processor wrapping the HF ``Step3VLProcessor`` (remote code).

    Produces a tokenized input stream with image-token placeholders rewritten
    to an out-of-vocab sentinel (``vocab_size + 1``) so ``fuse_input_embeds``
    can locate them by ``input_ids >= vocab_size`` -- mirroring the
    Qwen2.5-VL / LlavaNext pattern in TRT-LLM.  The structural special tokens
    (``<im_start>``, ``<im_end>``, ``<patch_start>``, ``<patch_end>``,
    ``<patch_newline>``) keep their original token ids; the language model
    embeds those normally.
    """

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: Optional[AutoTokenizer],
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        )
        # The Step3p7 checkpoint ships ``processing_step3.Step3VLProcessor`` as
        # a remote-code module, but does not register an ``AutoProcessor`` entry
        # nor a ``processor_config.json``. ``AutoProcessor.from_pretrained``
        # therefore falls back to a tokenizer-only path that silently drops
        # the image input. Load the remote class directly so images are
        # actually preprocessed.
        #
        # ``Step3VLProcessor`` expects a raw HF tokenizer (uses ``get_vocab``).
        # The runtime may hand us a ``TransformersTokenizer`` wrapper instead;
        # unwrap it before constructing the processor.
        hf_tokenizer = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        try:
            processor_cls = get_class_from_dynamic_module(
                "processing_step3.Step3VLProcessor",
                model_path,
            )
            self._processor = processor_cls(
                tokenizer=hf_tokenizer,
                chat_template=getattr(hf_tokenizer, "chat_template", None),
            )
        except Exception:
            logger.warning(
                "[Step3p7VL] Falling back to AutoProcessor; image inputs may not be processed."
            )
            self._processor = AutoProcessor.from_pretrained(
                model_path,
                use_fast=self.use_fast,
                trust_remote_code=trust_remote_code,
            )
        # The remote ``Step3VLProcessor`` exposes ``image_token_id`` via the
        # tokenizer vocabulary, but its getter calls ``get_vocab()`` which
        # raises ``NotImplementedError`` on tokenizers loaded without the
        # fast backend. Prefer the model config value, falling back to the
        # processor only when it is safely accessible.
        cfg_image_token_id = getattr(config, "image_token_id", None)
        if cfg_image_token_id is None:
            try:
                cfg_image_token_id = self._processor.image_token_id  # type: ignore[attr-defined]
            except Exception:
                cfg_image_token_id = 128001
        self._image_token_id = int(cfg_image_token_id)
        text_config = getattr(config, "text_config", config)
        self._dtype = getattr(text_config, "torch_dtype", torch.bfloat16)
        if isinstance(self._dtype, str):
            self._dtype = getattr(torch, self._dtype, torch.bfloat16)
        self._vocab_size = int(getattr(text_config, "vocab_size", 0))
        self._tllm_multimodal_token_id = self._vocab_size + 1

        hf_tok = getattr(self._tokenizer, "tokenizer", self._tokenizer)
        special_token_ids: List[int] = []
        for tok in ("<im_start>", "<im_end>", "<patch_start>", "<patch_end>", "<patch_newline>"):
            try:
                tok_id = hf_tok.convert_tokens_to_ids(tok)
            except Exception:
                tok_id = None
            unk_id = getattr(hf_tok, "unk_token_id", None)
            if tok_id is None or (unk_id is not None and tok_id == unk_id):
                logger.warning(
                    "[Step3p7VL] Could not resolve structural token %r; "
                    "multimodal hashing will fall back to the vocab-size "
                    "discriminator only.",
                    tok,
                )
                special_token_ids = []
                break
            special_token_ids.append(int(tok_id))
        self._mm_special_token_ids = special_token_ids

    # ------- BaseMultimodalInputProcessor required properties -------------

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # ----------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def get_num_tokens_per_image(
        self,
        *,
        image: Union[Image.Image, torch.Tensor],
        **kwargs,
    ) -> int:
        """Total prompt tokens for one image, framing tokens included.

        Delegates to the remote processor's ``get_num_image_tokens`` (which
        accounts for ``<patch_start>``/``<patch_end>``/``<patch_newline>`` per
        tile and ``<im_start>``/``<im_end>`` for the global features), so the
        count matches the contiguous span produced by ``call_with_text_prompt``.
        """
        if isinstance(image, torch.Tensor):
            height, width = int(image.shape[-2]), int(image.shape[-1])
        else:
            width, height = image.width, image.height
        return int(self._processor.get_num_image_tokens(width, height))

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        """Token ids forming one logical image unit (embed slots + framing).

        ``call_with_text_prompt`` rewrites the ``<im_patch>`` placeholders to the OOV
        sentinel ``vocab_size + 1``, so the embed slots are matched by that
        sentinel rather than the original ``<im_patch>`` id. Returning the
        sentinel together with the framing tokens keeps the whole image span
        contiguous in the hashing mask. Falls back to ``None`` (vocab-size
        discriminator) when the framing tokens could not be resolved.
        """
        if not self._mm_special_token_ids:
            return None
        return torch.tensor([self._tllm_multimodal_token_id] + self._mm_special_token_ids)

    def get_mm_special_token_ids(self) -> Optional[torch.Tensor]:
        """Framing-token ids inside an image span that carry no vision embed.

        These are subtracted from the embed-row mask so the embed-slot count
        stays accurate while the span itself remains contiguous.
        """
        if not self._mm_special_token_ids:
            return None
        return torch.tensor(self._mm_special_token_ids)

    @torch.inference_mode()
    def call_with_text_prompt(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data") or {}
        images = mm_data.get("image", []) if isinstance(mm_data, dict) else []

        if not images:
            token_ids = self._tokenizer(text_prompt, return_tensors="pt").input_ids[0]
            return token_ids.to(torch.int32).tolist(), {}

        processed = self._processor(
            text=text_prompt,
            images=images,
            return_tensors="pt",
        )
        input_ids = processed["input_ids"][0]

        # Replace each <im_patch> position with the OOV sentinel; structural
        # tokens (<im_start>, <patch_start>, etc.) keep their normal ids.
        sentinel = self._tllm_multimodal_token_id
        input_ids = input_ids.clone()
        input_ids[input_ids == self._image_token_id] = sentinel

        multimodal_data: Dict[str, Any] = {"image": {}}
        image_dict = multimodal_data["image"]
        image_dict["pixel_values"] = processed["pixel_values"].to(self._dtype)
        if "patch_pixel_values" in processed:
            image_dict["patch_pixel_values"] = processed["patch_pixel_values"].to(self._dtype)
        if "num_patches" in processed:
            np_val = processed["num_patches"]
            image_dict["num_patches"] = (
                np_val.tolist() if isinstance(np_val, torch.Tensor) else list(np_val)
            )
        if "patch_newline_mask" in processed:
            image_dict["patch_newline_mask"] = processed["patch_newline_mask"]

        return input_ids.to(torch.int32).tolist(), {"multimodal_data": multimodal_data}


# ---------------------------------------------------------------------------
# VLM wrapper (vision + language)
# ---------------------------------------------------------------------------


@register_vision_encoder(Step3p7VisionTower)
@register_auto_model("Step3p7ForConditionalGeneration")
@register_input_processor(
    Step3p7VLInputProcessor,
    model_type="step3p7",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<im_patch>"},
        content_format=ContentFormat.OPENAI,
    ),
)
class Step3p7VLForConditionalGeneration(nn.Module):
    """Multimodal entry point for the Step3p7 Flash checkpoint.

    Wraps the existing text-only ``Step3p7ForCausalLM`` plus the
    PerceptionEncoder-based vision tower.  When a request has no
    multimodal payload this class is a thin passthrough so plain text
    benchmarks continue to use the original Step3p7 text path.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config

        # Mirror text-config aliases up onto the top-level config (mirrors
        # what Step3p7ForCausalLM.__init__ does) before instantiating the
        # text model; otherwise the inner __init__ would be working on a
        # config whose torch_dtype etc. are still strings.
        _mirror_step3p7_text_aliases(self.config)

        # Inner causal LM: reuses all the existing decoder + MTP wiring.
        self.llm = Step3p7ForCausalLM(model_config)

        # Vision encoder. Built lazily in load_weights() (outside MetaInitMode)
        # so its PerceptionEncoder / HF submodules allocate real tensors instead
        # of meta tensors. This keeps the large text LLM on the fast meta-init
        # path while avoiding leftover meta tensors that would crash at runtime.
        # Stays None in the disaggregated decode worker.
        self.mm_encoder = None

    # ----- engine-facing surface (delegate to the inner causal LM) -------

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @property
    def model(self):
        # Expose the inner decoder model so existing helpers that walk
        # ``self.model.layers`` (e.g. weight bookkeeping) keep working when
        # they see the wrapper instead of the bare causal LM.
        return self.llm.model

    @property
    def lm_head(self):
        return self.llm.lm_head

    @property
    def epilogue(self):
        return self.llm.epilogue

    @property
    def spec_worker(self):
        return getattr(self.llm, "spec_worker", None)

    @property
    def draft_model(self):
        return getattr(self.llm, "draft_model", None)

    # ---------------------------------------------------------------------

    def load_weights(
        self,
        weights,
        weight_mapper=None,
        skip_modules=None,
        params_map=None,
        allow_partial_loading: bool = False,
    ):
        """Split vision/text weights and delegate to the inner LM loader."""
        if self.mm_encoder is None and not _is_mm_disagg() and hasattr(weights, "items"):
            # Construct the vision tower here, outside MetaInitMode, so its
            # PerceptionEncoder / HF submodules allocate real tensors. Move it
            # straight to CUDA (model_loader already ran model.to("cuda") for
            # the LLM); the load_state_dict below copies the checkpoint in.
            self.mm_encoder = Step3p7VisionTower(self.model_config).eval().to("cuda")
        if self.mm_encoder is not None and hasattr(weights, "items"):
            # Hand the vision subtree to the encoder; it consumes the keys
            # in-place via state_dict (no removal from ``weights`` needed —
            # ``Step3p7ForCausalLM.load_weights`` already strips
            # ``vision_model.`` / ``vit_large_projector.`` via
            # ``ignored_key_prefixes``).
            self.mm_encoder.load_weights(weights)
        return self.llm.load_weights(
            weights,
            weight_mapper=weight_mapper,
            skip_modules=skip_modules,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        spec_metadata: Optional[SpecMetadata] = None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_params = kwargs.pop("multimodal_params", [])
        num_context_requests = attn_metadata.num_contexts

        mm_embeds: List[torch.Tensor] = []
        # Only context requests carry pixel values; generation steps after
        # the first iteration have no image payload to encode.
        mm_context_params = [
            p
            for p in multimodal_params[:num_context_requests]
            if (
                p.multimodal_data is not None
                and (
                    p.multimodal_data.get("image", {}).get("pixel_values") is not None
                    or p.multimodal_data.get("multimodal_embedding") is not None
                )
            )
        ]
        if mm_context_params and self.mm_encoder is not None:
            mm_embeds = get_multimodal_embeddings(
                encoder_forward_fn=self.mm_encoder.forward,
                multimodal_params=mm_context_params,
            )
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_context_params)

        if input_ids is not None and mm_embeds:
            input_ids, inputs_embeds = fuse_input_embeds(
                self.llm.model.embed_tokens, input_ids, mm_embeds, **kwargs
            )

        return self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            spec_metadata=spec_metadata,
            resource_manager=resource_manager,
            **kwargs,
        )
