# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-Image MMDiT transformer.

Pure-PyTorch port of the reference implementation in
``diffusers.models.transformers.transformer_qwenimage`` at commit
matching diffusers 0.37.1. Module and attribute names mirror the
diffusers reference exactly so the HuggingFace checkpoint's
``transformer/*.safetensors`` state_dict can be loaded verbatim.

The current implementation prioritizes checkpoint compatibility and
diffusers parity. Performance-oriented swaps to TensorRT-LLM modules can
be made incrementally while preserving the public module names used by
checkpoint loading.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader

_WEIGHT_KEY_REMAPS = [
    (".net.0.proj.", ".up_proj."),
    (".net.2.", ".down_proj."),
]


def _remap_checkpoint_keys(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped = {}
    for key, value in weights.items():
        new_key = key
        for old, new in _WEIGHT_KEY_REMAPS:
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


# ===========================================================================
# Sinusoidal timestep embedding utilities.
# ===========================================================================


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal positional embedding, bit-exact port of diffusers."""
    if len(timesteps.shape) != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0,
        end=half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Stateless sinusoidal timestep feature extractor."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Two-linear timestep-to-conditioning projector (SiLU between)."""

    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class QwenTimestepProjEmbeddings(nn.Module):
    """Qwen-Image timestep-conditioning stack."""

    def __init__(self, embedding_dim: int, use_additional_t_cond: bool = False):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_states: torch.Tensor,
        addition_t_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))
        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError(
                    "When use_additional_t_cond=True, addition_t_cond must be provided."
                )
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb
        return conditioning


class AdaLayerNormContinuous(nn.Module):
    """AdaLN continuous: SiLU -> Linear(cond, 2*dim) -> LayerNorm(x)*scale+shift.

    Matches ``diffusers.models.normalization.AdaLayerNormContinuous``
    for Qwen-Image config: ``elementwise_affine=False``, ``eps=1e-6``,
    ``norm_type="layer_norm"``.
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        bias: bool = True,
        norm_type: str = "layer_norm",
        dtype: torch.dtype = None,
        quant_config=None,
        skip_create_weights: bool = False,
        force_dynamic_quant: bool = False,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = Linear(
            conditioning_embedding_dim,
            embedding_dim * 2,
            bias=bias,
            dtype=dtype,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
        )
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(hidden_size=embedding_dim, eps=eps, has_weights=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


# ===========================================================================
# Feed-forward.
# ===========================================================================


@torch.compiler.disable
def _gelu_tanh_eager(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class FeedForward(MLP):
    """TRT-LLM MLP with Qwen-Image GELU activation."""

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        activation_fn: str = "gelu-approximate",
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        config: Optional[DiffusionModelConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        if activation_fn == "gelu-approximate":
            activation = _gelu_tanh_eager
        elif activation_fn == "gelu":
            activation = F.gelu
        else:
            raise ValueError(
                f"Unsupported activation_fn={activation_fn} in Qwen-Image "
                "FeedForward; only gelu / gelu-approximate needed."
            )
        if dim_out != dim:
            raise ValueError("TRT-LLM MLP FeedForward requires dim_out == dim")
        super().__init__(
            hidden_size=dim,
            intermediate_size=inner_dim,
            bias=bias,
            activation=activation,
            dtype=dtype,
            config=config,
            layer_idx=layer_idx,
            reduce_output=False,
        )


# ===========================================================================
# 3D rotary position embedding.
# ===========================================================================


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    use_real: bool = False,
) -> torch.Tensor:
    """Apply complex-valued rotary embeddings; use_real=False is Qwen-Image.

    Bit-exact port of ``apply_rotary_emb_qwen`` from diffusers with
    ``use_real=False`` and ``use_real_unbind_dim=-1``.
    """
    if use_real:
        raise NotImplementedError(
            "Qwen-Image uses complex-valued freqs (use_real=False); the real-valued path is unused."
        )
    # The shared VisualGen apply_rotary_emb helper takes real-valued
    # cos/sin tensors. Qwen-Image stores RoPE as complex frequencies to
    # match diffusers, so keep this adapter until we add a shared complex
    # RoPE helper or normalize Qwen's cache format.
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
    return x_out.type_as(x)


class QwenEmbedRope(nn.Module):
    """3D rotary position embedding over (frame, height, width) axes.

    Identical math and memory layout as diffusers'
    ``transformer_qwenimage.QwenEmbedRope``. Stores the pos/neg complex
    frequency buffers as regular tensor attributes (not ``nn.Buffer``)
    to preserve the imaginary part, just like diffusers does. Text RoPE
    is a slice of ``pos_freqs`` offset by the max video index.

    Note on caching: the ``functools.lru_cache`` decorator on
    ``_compute_video_freqs`` is keyed on ``self`` via the ``self`` arg,
    which makes it a per-instance cache. That matches diffusers'
    behaviour exactly.
    """

    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self._rope_params(pos_index, axes_dim[0], theta),
                self._rope_params(pos_index, axes_dim[1], theta),
                self._rope_params(pos_index, axes_dim[2], theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self._rope_params(neg_index, axes_dim[0], theta),
                self._rope_params(neg_index, axes_dim[1], theta),
                self._rope_params(neg_index, axes_dim[2], theta),
            ],
            dim=1,
        )
        # Intentionally not registered as buffers: register_buffer would
        # drop the imaginary part on dtype conversion. Matches diffusers.
        self.scale_rope = scale_rope

    @staticmethod
    def _rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )
        return torch.polar(torch.ones_like(freqs), freqs)

    @functools.lru_cache(maxsize=8)
    def _pos_freqs_for_device(self, device: Optional[torch.device]) -> torch.Tensor:
        if device is None:
            return self.pos_freqs
        return self.pos_freqs.to(device)

    def forward(
        self,
        video_fhw,
        max_txt_seq_len: int | torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx, device)
            vid_freqs.append(video_freq)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_txt_seq_len_int = int(max_txt_seq_len)
        txt_freqs = self._pos_freqs_for_device(device)[
            max_vid_index : max_vid_index + max_txt_seq_len_int, ...
        ]
        vid_freqs = torch.cat(vid_freqs, dim=0)
        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=128)
    def _compute_video_freqs(
        self,
        frame: int,
        height: int,
        width: int,
        idx: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        pos_freqs = self.pos_freqs.to(device) if device is not None else self.pos_freqs
        neg_freqs = self.neg_freqs.to(device) if device is not None else self.neg_freqs

        freqs_pos = pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = (
            freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        )
        if self.scale_rope:
            freqs_height = torch.cat(
                [freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]],
                dim=0,
            )
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat(
                [freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]],
                dim=0,
            )
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = (
                freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            )
            freqs_width = (
                freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            )

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


# ===========================================================================
# Joint self-attention for MMDiT.
# ===========================================================================


class QwenJointAttention(Attention):
    """Double-stream joint attention.

    Holds the image-stream (to_q/k/v), text-stream (add_q/k/v_proj),
    per-stream QK-norms, and the two output projections (to_out.0 for
    image, to_add_out for text) as direct submodules so the HF state_dict
    loads with the checkpoint remapping in ``load_weights``. The common
    unmasked path uses the VisualGen attention backend; the masked path
    falls back to torch SDPA because VisualGen's backend mask contract is
    currently limited to predefined masks.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
        config: Optional[DiffusionModelConfig] = None,
        layer_idx: int = 0,
    ):
        config = config or DiffusionModelConfig()
        super().__init__(
            hidden_size=dim,
            num_attention_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qkv_mode=QKVMode.SEPARATE_QKV,
            qk_norm=True,
            qk_norm_mode="per_head",
            eps=eps,
            bias=True,
            # TODO: enable fused qk-norm+RoPE after adapting Qwen's
            # complex frequency cache to the shared real cos/sin format.
            fuse_qk_norm_rope=False,
            config=config,
            layer_idx=layer_idx,
        )
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim

        # Text-stream QKV (diffusers names).
        self.add_q_proj = Linear(
            dim,
            dim,
            bias=True,
            dtype=dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )
        self.add_k_proj = Linear(
            dim,
            dim,
            bias=True,
            dtype=dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )
        self.add_v_proj = Linear(
            dim,
            dim,
            bias=True,
            dtype=dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )

        # QK-norms, applied per-head on the head_dim.
        self.norm_added_q = RMSNorm(
            hidden_size=attention_head_dim, eps=eps, dtype=dtype, has_weights=True
        )
        self.norm_added_k = RMSNorm(
            hidden_size=attention_head_dim, eps=eps, dtype=dtype, has_weights=True
        )
        # Text output projection.
        self.to_add_out = Linear(
            dim,
            dim,
            bias=True,
            dtype=dtype,
            mapping=self.mapping,
            quant_config=self.quant_config,
            skip_create_weights_in_init=self.skip_create_weights_in_init,
            force_dynamic_quantization=self.force_dynamic_quantization,
        )

    @staticmethod
    def _apply_rms_norm(x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), norm.weight, norm.variance_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_txt = encoder_hidden_states.shape[1]

        # Image QKV.
        img_q, img_k, img_v = self.get_qkv(hidden_states)
        # Text QKV.
        txt_q = self.add_q_proj(encoder_hidden_states)
        txt_k = self.add_k_proj(encoder_hidden_states)
        txt_v = self.add_v_proj(encoder_hidden_states)

        # Reshape to (B, S, H, D).
        img_q = img_q.unflatten(-1, (self.heads, -1))
        img_k = img_k.unflatten(-1, (self.heads, -1))
        img_v = img_v.unflatten(-1, (self.heads, -1))
        txt_q = txt_q.unflatten(-1, (self.heads, -1))
        txt_k = txt_k.unflatten(-1, (self.heads, -1))
        txt_v = txt_v.unflatten(-1, (self.heads, -1))

        # Per-stream QK-norm on head dim.
        img_q = self._apply_rms_norm(img_q, self.norm_q)
        img_k = self._apply_rms_norm(img_k, self.norm_k)
        txt_q = self._apply_rms_norm(txt_q, self.norm_added_q)
        txt_k = self._apply_rms_norm(txt_k, self.norm_added_k)

        # Rotary on both streams.
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = apply_rotary_emb_qwen(img_q, img_freqs, use_real=False)
            img_k = apply_rotary_emb_qwen(img_k, img_freqs, use_real=False)
            txt_q = apply_rotary_emb_qwen(txt_q, txt_freqs, use_real=False)
            txt_k = apply_rotary_emb_qwen(txt_k, txt_freqs, use_real=False)

        # Joint attention: [txt | img] order.
        joint_q = torch.cat([txt_q, img_q], dim=1)
        joint_k = torch.cat([txt_k, img_k], dim=1)
        joint_v = torch.cat([txt_v, img_v], dim=1)

        # SDPA expects (B, H, S, D); diffusers dispatch_attention_fn
        # accepts (B, S, H, D) and transposes internally. Do the same.
        joint_q = joint_q.transpose(1, 2)
        joint_k = joint_k.transpose(1, 2)
        joint_v = joint_v.transpose(1, 2)

        attn_mask = None
        if attention_mask is not None:
            # attention_mask is (B, Sjoint) bool or float. Expand to
            # (B, 1, 1, Sjoint) so SDPA broadcasts over (H, Sq).
            # Qwen pads text embeddings before concatenating [text | image],
            # so masked SDPA is required to ignore padded text tokens.
            attn_mask = attention_mask[:, None, None, :]

        if attn_mask is None:
            out = self._attn_impl(
                joint_q.transpose(1, 2).flatten(2),
                joint_k.transpose(1, 2).flatten(2),
                joint_v.transpose(1, 2).flatten(2),
            )
        else:
            out = F.scaled_dot_product_attention(
                joint_q, joint_k, joint_v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
            )
            out = out.transpose(1, 2).flatten(2, 3).to(joint_q.dtype)

        txt_attn_output = out[:, :seq_txt, :]
        img_attn_output = out[:, seq_txt:, :]

        img_attn_output = self.to_out[0](img_attn_output.contiguous())
        txt_attn_output = self.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


# ===========================================================================
# MMDiT double-stream block.
# ===========================================================================


class QwenImageTransformerBlock(nn.Module):
    """One layer of the Qwen-Image MMDiT stack."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
        config: Optional[DiffusionModelConfig] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        config = config or DiffusionModelConfig()

        # Image stream.
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            Linear(
                dim,
                6 * dim,
                bias=True,
                dtype=dtype,
                quant_config=config.get_quant_config(),
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
            ),
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenJointAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            eps=eps,
            dtype=dtype,
            config=config,
            layer_idx=layer_idx,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            dtype=dtype,
            config=config,
            layer_idx=layer_idx,
        )

        # Text stream (shares attention with image stream).
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            Linear(
                dim,
                6 * dim,
                bias=True,
                dtype=dtype,
                quant_config=config.get_quant_config(),
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                force_dynamic_quantization=config.force_dynamic_quantization,
            ),
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu-approximate",
            dtype=dtype,
            config=config,
            layer_idx=layer_idx,
        )

    @staticmethod
    def _modulate(x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        gate = gate.unsqueeze(1)
        return x * (1 + scale) + shift, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod_params = self.img_mod(temb)
        txt_mod_params = self.txt_mod(temb)

        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)

        # Norm1 + modulation.
        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        # Joint attention.
        img_attn_output, txt_attn_output = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        # Residual.
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Norm2 + MLP + residual.
        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


# ===========================================================================
# Top-level transformer.
# ===========================================================================


class QwenImageTransformer2DModel(nn.Module):
    """Qwen-Image 20B MMDiT transformer.

    Mirrors ``diffusers.models.transformers.transformer_qwenimage.QwenImageTransformer2DModel``
    attribute-for-attribute so the HuggingFace ``transformer/*.safetensors``
    state_dict can be loaded via ``load_state_dict(strict=True)``.
    """

    def __init__(
        self,
        model_config: Optional["DiffusionModelConfig"] = None,
        *,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        attn_backend: str = "sdpa",
    ):
        super().__init__()
        self.model_config = model_config or DiffusionModelConfig()
        self.attn_backend = attn_backend

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)
        self.txt_norm = RMSNorm(
            hidden_size=joint_attention_dim,
            eps=1e-6,
            dtype=self.model_config.torch_dtype,
            has_weights=True,
        )
        linear_kwargs = {
            "dtype": self.model_config.torch_dtype,
            "quant_config": self.model_config.get_quant_config(),
            "skip_create_weights_in_init": self.model_config.skip_create_weights_in_init,
            "force_dynamic_quantization": self.model_config.force_dynamic_quantization,
        }
        self.img_in = Linear(in_channels, self.inner_dim, bias=True, **linear_kwargs)
        self.txt_in = Linear(joint_attention_dim, self.inner_dim, bias=True, **linear_kwargs)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dtype=self.model_config.torch_dtype,
                    config=self.model_config,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            dtype=self.model_config.torch_dtype,
            quant_config=self.model_config.get_quant_config(),
            skip_create_weights=self.model_config.skip_create_weights_in_init,
            force_dynamic_quant=self.model_config.force_dynamic_quantization,
        )
        self.proj_out = Linear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
            **linear_kwargs,
        )

    @property
    def device(self) -> torch.device:
        return self.proj_out.weight.device

    def _weight_loading_device(self) -> torch.device:
        for param in self.parameters():
            if param.device.type != "meta":
                return param.device
        for buffer in self.buffers():
            if buffer.device.type != "meta":
                return buffer.device
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    @classmethod
    def from_config_dict(cls, cfg: Dict[str, Any], **kwargs) -> "QwenImageTransformer2DModel":
        """Build from a transformer/config.json dict."""
        return cls(
            patch_size=cfg.get("patch_size", 2),
            in_channels=cfg.get("in_channels", 64),
            out_channels=cfg.get("out_channels", 16),
            num_layers=cfg.get("num_layers", 60),
            attention_head_dim=cfg.get("attention_head_dim", 128),
            num_attention_heads=cfg.get("num_attention_heads", 24),
            joint_attention_dim=cfg.get("joint_attention_dim", 3584),
            axes_dims_rope=tuple(cfg.get("axes_dims_rope", [16, 56, 56])),
            **kwargs,
        )

    def to_inference_dtype(self) -> "QwenImageTransformer2DModel":
        """Cast non-quantized parameters to the configured inference dtype.

        A blanket ``module.to(torch.bfloat16)`` breaks quantized TRT-LLM
        Linear modules by converting FP8/NVFP4 weights and FP32 scales.
        """
        target_dtype = self.model_config.torch_dtype
        for module in self.modules():
            if isinstance(module, Linear):
                continue
            for param in module.parameters(recurse=False):
                if param is not None and param.is_floating_point():
                    param.data = param.data.to(target_dtype)
            for buffer in module.buffers(recurse=False):
                if buffer.is_floating_point():
                    buffer.data = buffer.data.to(target_dtype)
        return self

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load HF ``transformer/*.safetensors`` state_dict.

        Feed-forward weights are remapped from diffusers' ``net.0`` /
        ``net.2`` names into the TRT-LLM ``MLP`` layout.
        """
        weights = _remap_checkpoint_keys(weights)

        device = self._weight_loading_device()
        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()
                module.to(device)

        expected = {name for name, _ in self.named_parameters()}
        provided = set(weights)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        # Dynamic quantization creates scale parameters while loading Linear
        # modules, so those keys are expected to be absent from BF16 checkpoints.
        if missing and not self.model_config.dynamic_weight_quant:
            raise RuntimeError(f"Missing keys when loading transformer: {missing[:5]}...")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading transformer: {unexpected[:5]}...")

        loader = DynamicLinearWeightLoader(self.model_config)
        for name, module in self.named_modules():
            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)
                if weight_dicts:
                    try:
                        loader.load_linear_weights(module, name, weight_dicts)
                    except Exception as exc:
                        src_weight = weight_dicts[0].get("weight")
                        src_shape = tuple(src_weight.shape) if src_weight is not None else None
                        src_dtype = src_weight.dtype if src_weight is not None else None
                        raise RuntimeError(
                            "Failed loading Qwen-Image Linear "
                            f"{name}: target_weight_shape={tuple(module.weight.shape)}, "
                            f"target_weight_dtype={module.weight.dtype}, "
                            f"source_weight_shape={src_shape}, "
                            f"source_weight_dtype={src_dtype}"
                        ) from exc
            else:
                module_weights = loader.filter_weights(name, weights)
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(module_weights[param_name].to(param.dtype))

    def post_load_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, Linear):
                try:
                    module.post_load_weights()
                except Exception as exc:
                    weight = getattr(module, "weight", None)
                    weight_scale = getattr(module, "weight_scale", None)
                    weight_shape = tuple(weight.shape) if weight is not None else None
                    weight_dtype = weight.dtype if weight is not None else None
                    scale_shape = tuple(weight_scale.shape) if weight_scale is not None else None
                    scale_dtype = weight_scale.dtype if weight_scale is not None else None
                    raise RuntimeError(
                        "Failed post_load_weights for Qwen-Image Linear "
                        f"{name}: quant_method={type(module.quant_method).__name__}, "
                        f"weight_shape={weight_shape}, "
                        f"weight_dtype={weight_dtype}, "
                        f"weight_scale_shape={scale_shape}, "
                        f"weight_scale_dtype={scale_dtype}"
                    ) from exc

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        img_shapes: Optional[list] = None,
        txt_seq_lens: Optional[list] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        del kwargs, txt_seq_lens  # Only kept for diffusers API compat.
        missing = []
        if timestep is None:
            missing.append("timestep")
        if img_shapes is None:
            missing.append("img_shapes")
        if missing:
            raise ValueError(f"Missing required argument(s): {', '.join(missing)}")

        # Project image tokens into model space.
        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        # Project text tokens.
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        text_seq_len = encoder_hidden_states.shape[1]

        # Timestep-conditioning vector.
        temb = self.time_text_embed(timestep, hidden_states)

        image_rotary_emb = self.pos_embed(
            img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device
        )

        # Build joint attention mask [text_mask | all-ones image_mask] once.
        block_attention_mask = None
        if encoder_hidden_states_mask is not None:
            if encoder_hidden_states_mask.dtype != torch.bool:
                encoder_hidden_states_mask = encoder_hidden_states_mask.to(torch.bool)
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones(
                (batch_size, image_seq_len),
                dtype=torch.bool,
                device=hidden_states.device,
            )
            block_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=block_attention_mask,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if return_dict:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=output)
        return (output,)
