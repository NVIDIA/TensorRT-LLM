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
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.mapping import Mapping

from ..._utils import nvtx_range, prefer_pinned
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
from ..attention_backend.interface import PredefinedAttentionMask
from ..attention_backend.utils import get_attention_backend
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear
from ..modules.mlp import MLP
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

    def freqs_for_grid(
        self,
        grid_hw: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Per-token 2D-RoPE frequencies for one image, shape ``(seq, dim)``.

        ``seq == grid_h * grid_w`` (plus one for a leading CLS token when
        ``use_cls_token``). Used by the encoder to build a flat frequency
        tensor over the concatenated varlen token stream.
        """
        if grid_hw[0] != self.max_grid_height or grid_hw[1] != self.max_grid_width:
            rows = torch.arange(grid_hw[0], device=device).view(-1, 1)
            cols = torch.arange(grid_hw[1], device=device).view(1, -1)
            positions = (rows * self.max_grid_width + cols).reshape(-1).to(torch.long)
            if self.use_cls_token:
                positions = torch.cat(
                    [
                        torch.zeros(1, device=device, dtype=torch.long),
                        positions + 1,
                    ],
                    dim=0,
                )
            freqs = self.freqs_cache.index_select(2, positions)
        else:
            freqs = self.freqs_cache
        # freqs_cache is (1, 1, seq, dim); drop the leading broadcast axes.
        return freqs.to(device)[0, 0]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Retained for the (B, heads, seq, dim) reference path used in tests.
        freqs = self.freqs_for_grid(grid_hw, q.device)[None, None]
        return _apply_rotary_emb(freqs, q), _apply_rotary_emb(freqs, k)


class Step3VisionLayerScale(nn.Module):
    """Per-channel residual scaling used when ``ls_init_value`` is set."""

    def __init__(self, dim: int, init_value: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.gamma


class Step3VisionMLP(MLP):
    """``c_fc -> act -> c_proj`` FFN, dispatched through TRT-LLM ``MLP``.

    The non-gated vision FFN maps directly onto the base ``MLP`` module
    (``up_proj -> activation -> down_proj``). The HF ``mlp.c_fc`` / ``mlp.c_proj``
    weights are remapped onto the base class's ``up_proj`` / ``down_proj`` in
    ``Step3p7VisionTower._remap_vision_weights``. The vision tower runs
    single-rank (``Mapping(world_size=1)``), so the base module's column/row
    tensor-parallel modes are no-ops here.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        dtype: torch.dtype,
    ):
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=True,
            activation=ACT2FN[hidden_act],
            dtype=dtype,
            config=model_config,
        )


# trtllm-gen FMHA ships cubins for these head_dim sizes only. The
# PerceptionEncoder head_dim (1536/16 = 96) is not in the set, so q/k/v/o_proj
# head dims are zero-padded up to the next supported size (128). The kernel
# sees zero-padded channels while 2D RoPE math runs on the real channels; the
# softmax scale is preserved through a compensating ``q_scaling`` (see below).
#
# TODO(perf): padding 96->128 wastes ~33% of the QK^T / P*V flops and adds a
# per-forward ``torch.cat`` on q/k/v every layer. The head_dim restriction is
# specific to the trtllm-gen cubins, not to attention in general: FlashInfer ragged
# prefill supports head_dim=96 natively. Dispatching the vision tower's
# attention through a head_dim-96-native backend would drop the padding, the
# ``q_scaling`` compensation, and the per-forward pads. The same hack exists in
# ``modeling_gemma4_vision.py`` (72->80), so a shared helper could fix both.
_FMHA_SUPPORTED_HEAD_DIMS = (64, 80, 128, 256, 512)


def _fmha_padded_head_dim(head_dim: int) -> int:
    if head_dim in _FMHA_SUPPORTED_HEAD_DIMS:
        return head_dim
    return next(d for d in _FMHA_SUPPORTED_HEAD_DIMS if d >= head_dim)


class Step3VisionAttention(Attention):
    """Vision MHA with 2D RoPE, dispatched through TRT-LLM ``Attention``.

    Subclasses ``Attention`` to participate in the TRT-LLM backend dispatch
    (context-only, ``PredefinedAttentionMask.FULL``, varlen via per-image
    ``attn_metadata``). The HF fused ``in_proj_*`` / ``out_proj`` weights are
    remapped onto the base class's fused ``qkv_proj`` / ``o_proj`` in
    ``Step3p7VisionTower.load_weights`` (with head_dim zero-padding).

    HF uses softmax ``scale = head_dim ** -0.5``. TRT-LLM uses
    ``qk_scale = 1 / (sqrt(self.head_dim) * q_scaling)`` with ``self.head_dim``
    being the *padded* size, so we pass ``q_scaling = sqrt(hf_head_dim /
    padded_head_dim)`` to recover ``1 / sqrt(hf_head_dim)`` exactly.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hidden_size: int,
        num_heads: int,
        layer_idx: int,
        dtype: torch.dtype,
        attn_bias: bool = True,
    ):
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )
        hf_head_dim = hidden_size // num_heads
        padded_head_dim = _fmha_padded_head_dim(hf_head_dim)
        q_scaling = math.sqrt(hf_head_dim / padded_head_dim)
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,  # vision MHA
            max_position_embeddings=None,
            bias=attn_bias,
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=dtype,
            dense_bias=attn_bias,
            config=model_config,
            q_scaling=q_scaling,
            head_dim=padded_head_dim,
        )
        # ``self.head_dim`` is the kernel-facing padded size; ``hf_head_dim`` is
        # the real width seen by 2D RoPE; ``head_dim_pad`` is the zero width.
        self.hf_head_dim = hf_head_dim
        self.head_dim_pad = padded_head_dim - hf_head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Flat (num_tokens, hidden) layout; per-image segments are described by
        # ``attn_metadata`` (FULL mask, context-only). ``freqs`` is the flat
        # per-token 2D-RoPE frequency tensor (num_tokens, 1, hf_head_dim).
        num_tokens = hidden_states.shape[0]
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_heads, self.head_dim)
        v = v.view(num_tokens, self.num_heads, self.head_dim)

        # Slice the real (unpadded) channels for RoPE so the rotation matches HF.
        if self.head_dim_pad > 0:
            q_real = q[..., : self.hf_head_dim]
            k_real = k[..., : self.hf_head_dim]
            v_real = v[..., : self.hf_head_dim]
        else:
            q_real, k_real, v_real = q, k, v

        if freqs is not None:
            q_real = _apply_rotary_emb(freqs, q_real)
            k_real = _apply_rotary_emb(freqs, k_real)

        # Re-pad with zeros for the FMHA kernel. Zeros in q/k pad don't change
        # QK^T; zeros in v pad produce zero output channels stripped by o_proj.
        if self.head_dim_pad > 0:
            pad_shape = q_real.shape[:-1] + (self.head_dim_pad,)
            q = torch.cat([q_real, q_real.new_zeros(pad_shape)], dim=-1)
            k = torch.cat([k_real, k_real.new_zeros(pad_shape)], dim=-1)
            v = torch.cat([v_real, v_real.new_zeros(pad_shape)], dim=-1)
        else:
            q, k, v = q_real, k_real, v_real

        q = q.reshape(num_tokens, self.num_heads * self.head_dim)
        k = k.reshape(num_tokens, self.num_heads * self.head_dim)
        v = v.reshape(num_tokens, self.num_heads * self.head_dim)

        # Keep q/k/v in the projection weight dtype so the FMHA dispatcher does
        # not fall back to the unfused path (RoPE can promote to fp32).
        target_dtype = self.qkv_proj.weight.dtype
        q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)

        q, k, v = self.convert_qkv(q, k, v)
        attn_output = self.forward_impl(
            q=q,
            k=k,
            v=v,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
            attention_window_size=None,
            attention_mask_data=None,
            mrope_config=None,
            attention_sinks=None,
        )
        return self.o_proj(attn_output, layer_idx=self.layer_idx)


class Step3VisionBlock(nn.Module):
    """Single vision transformer block (Pre-LN + LayerScale).

    Operates on a flat ``(num_tokens, hidden)`` stream; the 2D-RoPE
    frequencies (``freqs``) and ``attn_metadata`` are owned by the encoder and
    threaded through unchanged.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        hidden_act: str,
        layer_norm_eps: float,
        ls_init_value: Optional[float],
        dtype: torch.dtype,
    ):
        super().__init__()
        self.attn = Step3VisionAttention(
            model_config=model_config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_idx=layer_idx,
            dtype=dtype,
        )
        self.ln_1 = LayerNorm(hidden_size=hidden_size, eps=layer_norm_eps, dtype=dtype)
        self.ln_2 = LayerNorm(hidden_size=hidden_size, eps=layer_norm_eps, dtype=dtype)
        self.mlp = Step3VisionMLP(
            model_config=model_config,
            hidden_size=hidden_size,
            intermediate_size=int(hidden_size * mlp_ratio),
            hidden_act=hidden_act,
            dtype=dtype,
        )
        ls = ls_init_value if ls_init_value is not None else 1.0
        self.ls_1 = Step3VisionLayerScale(hidden_size, ls)
        self.ls_2 = Step3VisionLayerScale(hidden_size, ls)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(hidden_states, attn_metadata, freqs)
        hidden_states = residual + self.ls_1(hidden_states)

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.ls_2(hidden_states)
        return hidden_states


class Step3VisionTransformer(nn.Module):
    def __init__(self, depth: int, **block_kwargs):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [Step3VisionBlock(layer_idx=i, **block_kwargs) for i in range(depth)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        freqs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.resblocks:
            hidden_states = block(hidden_states, attn_metadata, freqs)
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

    def __init__(
        self,
        model_config: ModelConfig,
        vision_config: PretrainedConfig,
        dtype: torch.dtype,
    ):
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
            LayerNorm(hidden_size=self.hidden_size, eps=self.layer_norm_eps, dtype=dtype)
            if self.use_ln_pre
            else nn.Identity()
        )
        self.ln_post = (
            LayerNorm(hidden_size=self.hidden_size, eps=self.layer_norm_eps, dtype=dtype)
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
            model_config=model_config,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            ls_init_value=self.ls_init_value,
            dtype=dtype,
        )

        # 2D RoPE owned by the encoder; one cache shared across all layers
        # (frequencies depend only on the grid + head_dim). RoPE rotates the
        # real (unpadded) head channels, so its ``dim`` is the HF head_dim.
        self.rope: Optional[Step3VisionRope2D] = None
        if self.use_rope2d:
            self.rope = Step3VisionRope2D(
                dim=self.hidden_size // self.num_heads,
                max_grid_height=grid_size,
                max_grid_width=grid_size,
                use_cls_token=self.use_cls_token,
                theta=float(getattr(vision_config, "rope_theta", 10000.0)),
                theta_rescale_factor=float(
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

        # Context-only attention metadata (no KV cache); rebuilt lazily when a
        # batch needs more token capacity than the current allocation.
        self._metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        self._attn_metadata: Optional[AttentionMetadata] = None
        self._attn_metadata_capacity = 0

    def _prepare_attn_metadata(self, seq_lens: List[int]) -> AttentionMetadata:
        total_tokens = int(sum(seq_lens))
        if self._attn_metadata is None or total_tokens > self._attn_metadata_capacity:
            capacity = max(total_tokens, 8192)
            self._attn_metadata = self._metadata_cls(
                max_num_requests=8192,
                max_num_tokens=capacity,
                kv_cache_manager=None,
            )
            self._attn_metadata_capacity = capacity
        md = self._attn_metadata
        batch_size = len(seq_lens)
        md.num_contexts = batch_size
        md.request_ids = list(range(1, batch_size + 1))
        md.prompt_lens = list(seq_lens)
        md.seq_lens = torch.tensor(seq_lens, dtype=torch.int, pin_memory=prefer_pinned())
        md.max_seq_len = max(seq_lens) if seq_lens else 0
        md.prepare()
        return md

    def _flat_freqs(
        self, grid_hw: Tuple[int, int], batch: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        """Flat per-token 2D-RoPE frequencies for ``batch`` same-grid images.

        Shape ``(batch * seq, 1, hf_head_dim)``; the head axis is left as 1 so
        it broadcasts over attention heads. All images in a single ``_encode``
        call share the grid, so the single-image frequencies are simply tiled.
        """
        if self.rope is None:
            return None
        freqs = self.rope.freqs_for_grid(grid_hw, device).unsqueeze(1)  # (seq, 1, hf_hd)
        if batch > 1:
            freqs = freqs.repeat(batch, 1, 1)
        return freqs

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

    def _embed(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Conv patch-embed + (optional) CLS token + abs posemb + pre-LN.

        Returns ``(hidden (B, P, D), grid_hw)`` where ``P`` includes the CLS
        token when ``use_cls_token``.
        """
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
        return hidden, (grid_h, grid_w)

    def forward_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Pre-downsample patch features ``(B, Gh*Gw, D)`` for a same-shape batch.

        The transformer runs over a flat varlen token stream: each image is one
        context segment (``PredefinedAttentionMask.FULL``) in a tower-local
        ``attn_metadata``; the segments share the grid so RoPE freqs are tiled.
        """
        hidden, grid_hw = self._embed(pixel_values)  # (B, P, D)
        bsz, num_tokens, hidden_dim = hidden.shape
        flat = hidden.reshape(bsz * num_tokens, hidden_dim)
        freqs = self._flat_freqs(grid_hw, bsz, flat.device)
        attn_metadata = self._prepare_attn_metadata([num_tokens] * bsz)
        flat = self.transformer(flat, attn_metadata, freqs)
        if self.use_ln_post:
            flat = self.ln_post(flat)
        hidden = flat.reshape(bsz, num_tokens, hidden_dim)
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

        # The vision tower is replicated on every rank (it is not
        # tensor-parallel sharded and is bf16 even in the FP8/NVFP4 text
        # checkpoints). Build a dedicated single-rank, quant-disabled
        # ModelConfig for the TRT-LLM ``Attention``/``Linear`` submodules so
        # they neither shard nor quantize, while still honouring the parent's
        # attention backend selection.
        vision_model_config = ModelConfig(
            pretrained_config=vision_config,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            attn_backend=getattr(model_config, "attn_backend", "TRTLLM"),
            skip_create_weights_in_init=False,
        )
        self.vision_model = Step3p7VisionEncoder(
            vision_model_config, vision_config, dtype=outer_dtype
        )

        proj_in = 4 * int(vision_config.width)
        proj_out = int(text_config.hidden_size)
        proj_bias = bool(getattr(pretrained_config, "projector_bias", False))
        # Projector ported to a TRT-LLM ``Linear`` (replicated, single rank).
        # The parameter names stay ``weight``/``bias`` so ``vit_large_projector.*``
        # weights load directly without remapping.
        self.vit_large_projector = Linear(
            proj_in,
            proj_out,
            bias=proj_bias,
            dtype=outer_dtype,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            skip_create_weights_in_init=False,
        )

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
            vision_state = self._remap_vision_weights(vision_state)
            self.vision_model.load_state_dict(vision_state, strict=True)
        if projector_state:
            self.vit_large_projector.load_state_dict(projector_state, strict=True)

    def _remap_vision_weights(
        self, vision_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Remap HF block weights onto the TRT-LLM ``Attention`` / ``MLP`` layout.

        Attention: HF stores per-block attention as ``attn.in_proj_{weight,bias}``
        (a single ``3H x H`` matrix) and ``attn.out_proj.{weight,bias}``. The
        ported ``Attention`` uses a fused ``attn.qkv_proj`` and ``attn.o_proj``.
        When the HF head_dim (e.g. 96) is not in the FMHA cubin set, the q/k/v
        and o_proj head dims are zero-padded up to the kernel-supported size
        (e.g. 128) per head — zeros in the appended channels are inert
        (``pad·pad = 0`` in QK^T, zero V channels give zero output channels that
        o_proj's zero-padded columns then ignore), so the math is unchanged.

        MLP: HF stores the non-gated FFN as ``mlp.c_fc.{weight,bias}`` /
        ``mlp.c_proj.{weight,bias}``; the ported ``MLP`` module uses
        ``mlp.up_proj`` / ``mlp.down_proj``. These are renamed verbatim (no
        padding — the FFN dims are kernel-agnostic).
        """
        num_heads = self.vision_model.num_heads
        hidden = self.vision_model.hidden_size
        hf_head_dim = hidden // num_heads
        padded_head_dim = _fmha_padded_head_dim(hf_head_dim)
        pad = padded_head_dim - hf_head_dim

        def _pad_out_rows(t: torch.Tensor, heads: int) -> torch.Tensor:
            # (heads * hf_head_dim, ...) -> (heads * padded_head_dim, ...) by
            # zero-padding the head_dim within each head.
            tail = t.shape[1:]
            t = t.view(heads, hf_head_dim, *tail)
            zeros = t.new_zeros(heads, pad, *tail)
            return torch.cat([t, zeros], dim=1).reshape(heads * padded_head_dim, *tail)

        remapped: Dict[str, torch.Tensor] = {}
        for key, val in vision_state.items():
            if key.endswith(".attn.in_proj_weight"):
                prefix = key[: -len("in_proj_weight")]
                q, k, v = val.chunk(3, dim=0)  # each (heads * hf_head_dim, H)
                if pad > 0:
                    q, k, v = (_pad_out_rows(t, num_heads) for t in (q, k, v))
                remapped[prefix + "qkv_proj.weight"] = torch.cat([q, k, v], dim=0)
            elif key.endswith(".attn.in_proj_bias"):
                prefix = key[: -len("in_proj_bias")]
                q, k, v = val.chunk(3, dim=0)
                if pad > 0:
                    q, k, v = (_pad_out_rows(t, num_heads) for t in (q, k, v))
                remapped[prefix + "qkv_proj.bias"] = torch.cat([q, k, v], dim=0)
            elif key.endswith(".attn.out_proj.weight"):
                prefix = key[: -len("out_proj.weight")]
                w = val  # (H, heads * hf_head_dim) -- pad the input dim per head.
                if pad > 0:
                    w = w.view(-1, num_heads, hf_head_dim)
                    zeros = w.new_zeros(w.shape[0], num_heads, pad)
                    w = torch.cat([w, zeros], dim=2).reshape(-1, num_heads * padded_head_dim)
                remapped[prefix + "o_proj.weight"] = w
            elif key.endswith(".attn.out_proj.bias"):
                prefix = key[: -len("out_proj.bias")]
                remapped[prefix + "o_proj.bias"] = val
            elif key.endswith(".mlp.c_fc.weight"):
                remapped[key[: -len("c_fc.weight")] + "up_proj.weight"] = val
            elif key.endswith(".mlp.c_fc.bias"):
                remapped[key[: -len("c_fc.bias")] + "up_proj.bias"] = val
            elif key.endswith(".mlp.c_proj.weight"):
                remapped[key[: -len("c_proj.weight")] + "down_proj.weight"] = val
            elif key.endswith(".mlp.c_proj.bias"):
                remapped[key[: -len("c_proj.bias")] + "down_proj.bias"] = val
            else:
                remapped[key] = val
        return remapped

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

        All images and patches across the whole batch are gathered, grouped by
        pixel-tensor shape, and encoded in as few batched vision passes as
        possible (one ``_encode`` per distinct shape), then scattered back and
        reassembled per request according to ``num_patches``.
        """
        device = self.vision_model.conv1.weight.device

        # ---- Pass 1: gather every image / patch across all requests. ----
        # ``order`` keeps request order; per request we hold its full images,
        # its patch images, and the per-full-image patch counts.
        order: List[int] = []
        full_imgs: Dict[int, List[torch.Tensor]] = {}
        patch_imgs: Dict[int, List[torch.Tensor]] = {}
        num_patches: Dict[int, List[int]] = {}

        def _flatten_to_images(t: torch.Tensor) -> List[torch.Tensor]:
            if t.dim() >= 5:
                t = t.view(-1, *t.shape[-3:])
            elif t.dim() == 3:
                t = t.unsqueeze(0)
            return list(t.to(device))

        for req_idx, mm in enumerate(multimodal_params):
            image_data = mm.multimodal_data.get("image") if mm.multimodal_data else None
            if image_data is None:
                continue
            pixel_values = image_data.get("pixel_values")
            if pixel_values is None:
                continue
            fulls = _flatten_to_images(pixel_values)

            patches: List[torch.Tensor] = []
            patch_pixel_values = image_data.get("patch_pixel_values")
            if patch_pixel_values is not None and patch_pixel_values.numel() > 0:
                patches = _flatten_to_images(patch_pixel_values)

            npl = image_data.get("num_patches")
            if npl is None:
                npl = [0] * len(fulls)
            elif isinstance(npl, torch.Tensor):
                npl = npl.flatten().tolist()
            else:
                npl = list(npl)

            order.append(req_idx)
            full_imgs[req_idx] = fulls
            patch_imgs[req_idx] = patches
            num_patches[req_idx] = npl

        if not order:
            return []

        # ---- Pass 2: group every image by pixel-tensor shape and encode each
        # group in a single batched vision pass; scatter features back. ----
        to_encode: List[Tuple[torch.Tensor, Tuple[int, str, int]]] = []
        for req_idx in order:
            for i, img in enumerate(full_imgs[req_idx]):
                to_encode.append((img, (req_idx, "full", i)))
            for i, img in enumerate(patch_imgs[req_idx]):
                to_encode.append((img, (req_idx, "patch", i)))

        shape_groups: Dict[Tuple[int, ...], List[int]] = {}
        for gi, (img, _ref) in enumerate(to_encode):
            shape_groups.setdefault(tuple(img.shape), []).append(gi)

        feats: List[Optional[torch.Tensor]] = [None] * len(to_encode)
        for idxs in shape_groups.values():
            batch = torch.stack([to_encode[gi][0] for gi in idxs], dim=0)
            encoded = self._encode(batch)  # (G, P, text_hidden)
            for j, gi in enumerate(idxs):
                feats[gi] = encoded[j]

        full_feats: Dict[int, List[torch.Tensor]] = {r: [None] * len(full_imgs[r]) for r in order}
        patch_feats: Dict[int, List[torch.Tensor]] = {r: [None] * len(patch_imgs[r]) for r in order}
        for gi, (_img, (req_idx, role, i)) in enumerate(to_encode):
            (full_feats if role == "full" else patch_feats)[req_idx][i] = feats[gi]

        # ---- Pass 3: reassemble each request as [patches... | full image]. ----
        per_request_embeds: List[torch.Tensor] = []
        for req_idx in order:
            ff = full_feats[req_idx]
            pf = patch_feats[req_idx]
            cur_patch_idx = 0
            flat_blocks: List[torch.Tensor] = []
            for img_idx, n_p in enumerate(num_patches[req_idx]):
                for j in range(cur_patch_idx, cur_patch_idx + n_p):
                    blk = pf[j]
                    flat_blocks.append(blk.reshape(-1, blk.shape[-1]))
                cur_patch_idx += n_p
                full_blk = ff[img_idx]
                flat_blocks.append(full_blk.reshape(-1, full_blk.shape[-1]))
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

        spec_input_ids = input_ids
        if input_ids is not None and mm_embeds:
            vocab_size = self.llm.model.embed_tokens.num_embeddings
            image_token_id = int(getattr(self.config, "image_token_id", 128001))
            spec_input_ids = torch.where(
                input_ids >= vocab_size,
                input_ids.new_full((), image_token_id),
                input_ids,
            )
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
            spec_input_ids=spec_input_ids,
            **kwargs,
        )
