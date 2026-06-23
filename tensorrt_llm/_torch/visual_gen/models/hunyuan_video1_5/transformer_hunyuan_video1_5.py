# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput

# Some existing Hunyuan components can be imported from HF, as they are trivial.
# However, HunyuanVideo15ImageProjection, HunyuanVideo15TokenRefiner, and
# HunyuanVideo15ByT5TextProjection must be reimplemented to avoid meta tensor errors.
from diffusers.models.transformers.transformer_hunyuan_video15 import (
    HunyuanVideo15PatchEmbed,
    HunyuanVideo15TimeEmbedding,
)
from tqdm import tqdm

from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
from tensorrt_llm.models.modeling_utils import QuantConfig

from .timestep_embedding import CombinedTimestepTextProjEmbeddings

_WEIGHT_KEY_REMAPS = [
    (".net.0.proj.", ".up_proj."),
    (".net.2.", ".down_proj."),
]


# Torch compiler disable is here based on Qwen Image
@torch.compiler.disable
def _gelu_tanh_eager(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


# Copied from HF
def _apply_rotary_emb(x: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor]):
    cos, sin = freqs_cis  # [S, D]
    cos = cos[None, :, None, :]  # sequence_dim=1 -> [1, S, 1, D]
    sin = sin[None, :, None, :]
    cos, sin = cos.to(x.device), sin.to(x.device)

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def _get_1d_rotary_pos_embed(
    dim: int,
    pos: torch.Tensor,
    theta: float = 10000.0,
    freqs_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dim % 2 == 0
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # [S, D/2]
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()
    return freqs_cos, freqs_sin


class HunyuanVideo15RotaryPosEmbed(torch.nn.Module):
    def __init__(
        self, patch_size: int, patch_size_t: int, rope_dim: List[int], theta: float = 256.0
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, height, width = hidden_states.shape
        rope_sizes = [
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        ]

        axes_grids = [
            torch.arange(0, size, device=hidden_states.device, dtype=torch.float32)
            for size in rope_sizes
        ]
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [T, H, W]
        grid = torch.stack(grid, dim=0)  # [3, T, H, W]

        freqs = [
            _get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta)
            for i in range(3)
        ]

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # [T * H * W, D / 2]
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # [T * H * W, D / 2]
        return freqs_cos, freqs_sin


class HunyuanVideo15FeedForward(MLP):
    """TRT-LLM MLP with HunyuanVideo1.5 GELU activation."""

    def __init__(
        self,
        dim: int,
        mult: float = 4.0,
        activation_fn: str = "gelu-approximate",
        dtype: Optional[torch.dtype] = None,
        model_config: Optional[DiffusionModelConfig] = None,
        layer_idx: Optional[int] = None,
    ) -> None:
        if activation_fn == "gelu-approximate":
            activation = _gelu_tanh_eager
        elif activation_fn == "linear-silu":
            activation = F.silu
        else:
            raise ValueError(
                f"Unsupported activation_fn={activation_fn} for HunyuanVideo15FeedForward; "
                "only 'gelu-approximate' and 'linear-silu' are used."
            )
        if dtype is None and model_config is not None:
            dtype = model_config.torch_dtype
        super().__init__(
            hidden_size=dim,
            intermediate_size=int(dim * mult),
            bias=True,
            activation=activation,
            dtype=dtype,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=False,
        )


def _aux_linear(
    in_features: int,
    out_features: int,
    bias: bool = True,
    model_config: Optional[DiffusionModelConfig] = None,
) -> Linear:
    """Build a quant-aware Linear for an auxiliary (non-block) projection."""
    return Linear(
        in_features,
        out_features,
        bias=bias,
        dtype=model_config.torch_dtype if model_config else None,
        quant_config=model_config.quant_config if model_config else None,
        skip_create_weights_in_init=(
            model_config.skip_create_weights_in_init if model_config else False
        ),
        force_dynamic_quantization=(
            model_config.force_dynamic_quantization if model_config else False
        ),
    )


def _remap_checkpoint_keys(weights: dict) -> dict:
    """Remap HuggingFace checkpoint keys to our module attribute names.

    HF diffusers uses nn.ModuleList wrappers that add numeric indices to
    weight key paths. Our simplified module structure uses plain attributes,
    so we translate the keys at load time.
    """
    remapped = {}
    for key, value in weights.items():
        new_key = key
        for old, new in _WEIGHT_KEY_REMAPS:
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


class HunyuanVideo15AdaLayerNormZero(torch.nn.Module):
    """TRT-LLM replacement for diffusers AdaLayerNormZero."""

    def __init__(
        self,
        embedding_dim: int,
        model_config: Optional[DiffusionModelConfig] = None,
    ) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = _aux_linear(embedding_dim, 6 * embedding_dim, model_config=model_config)

        # Kept elementwise_affine=False LayerNorms as torch equivalent
        self.norm = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class HunyuanVideo15AdaLayerNormContinuous(torch.nn.Module):
    """TRT-LLM replacement for diffusers AdaLayerNormContinuous."""

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = False,
        eps: float = 1e-6,
        model_config: Optional[DiffusionModelConfig] = None,
    ) -> None:
        super().__init__()
        self.silu = torch.nn.SiLU()
        self.linear = _aux_linear(
            conditioning_embedding_dim, embedding_dim * 2, model_config=model_config
        )
        self.norm = torch.nn.LayerNorm(
            embedding_dim, eps=eps, elementwise_affine=elementwise_affine
        )

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class HunyuanVideo15ImageProjection(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        # LayerNorm uses eps=1e-5 to match torch implementation
        self.norm_in = LayerNorm(hidden_size=in_channels, eps=1e-5)
        self.linear_1 = _aux_linear(in_channels, in_channels, model_config=model_config)
        self.act_fn = F.gelu
        self.linear_2 = _aux_linear(in_channels, hidden_size, model_config=model_config)
        self.norm_out = LayerNorm(hidden_size=hidden_size, eps=1e-5)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm_in(image_embeds)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


class HunyuanVideo15AdaNorm(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        model_config: Optional[DiffusionModelConfig] = None,
    ) -> None:
        super().__init__()

        out_features = out_features or 2 * in_features
        self.linear = _aux_linear(in_features, out_features, model_config=model_config)
        self.nonlinearity = F.silu

    def forward(
        self, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temb = self.linear(self.nonlinearity(temb))
        gate_msa, gate_mlp = temb.chunk(2, dim=1)
        gate_msa, gate_mlp = gate_msa.unsqueeze(1), gate_mlp.unsqueeze(1)
        return gate_msa, gate_mlp


class HunyuanVideo15RefinerAttention(Attention):
    """Token-refiner self-attention"""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        bias: bool = True,
        model_config: Optional[DiffusionModelConfig] = None,
    ) -> None:
        super().__init__(
            hidden_size=dim,
            num_attention_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qkv_mode=QKVMode.FUSE_QKV,
            qk_norm=False,
            bias=bias,
            config=model_config or DiffusionModelConfig(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = attention_mask.bool() if attention_mask is not None else None
        q, k, v = self.get_qkv(hidden_states)
        out = self._attn_impl(q, k, v, key_padding_mask=key_padding_mask)
        return self.to_out[0](out)


class HunyuanVideo15IndividualTokenRefinerBlock(torch.nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
        model_config: Optional[DiffusionModelConfig] = None,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        # has_weights and has_bias are equivalent to torch's elementwise_affine
        self.norm1 = LayerNorm(hidden_size=hidden_size, has_weights=True, has_bias=True, eps=1e-6)
        self.attn = HunyuanVideo15RefinerAttention(
            dim=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            bias=attention_bias,
            model_config=model_config,
        )

        self.norm2 = LayerNorm(hidden_size=hidden_size, has_weights=True, has_bias=True, eps=1e-6)
        self.ff = HunyuanVideo15FeedForward(
            hidden_size,
            mult=mlp_width_ratio,
            activation_fn="linear-silu",
            model_config=model_config,
        )

        self.norm_out = HunyuanVideo15AdaNorm(
            hidden_size, 2 * hidden_size, model_config=model_config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            attention_mask=attention_mask,
        )

        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_output * gate_msa

        ff_output = self.ff(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output * gate_mlp

        return hidden_states


class HunyuanVideo15IndividualTokenRefiner(torch.nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
        model_config: Optional[DiffusionModelConfig] = None,
    ) -> None:
        super().__init__()

        self.refiner_blocks = torch.nn.ModuleList(
            [
                HunyuanVideo15IndividualTokenRefinerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    attention_bias=attention_bias,
                    model_config=model_config,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        # The VisualGen attention backend masks padded keys via a [B, S]
        # key_padding_mask (valid query rows then attend only to valid keys,
        # matching the diffusers 2D mask for the tokens kept downstream).
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask.to(hidden_states.device).bool()

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, key_padding_mask)

        return hidden_states


class HunyuanVideo15TokenRefiner(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        attention_bias: bool = True,
        model_config: Optional[DiffusionModelConfig] = None,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=hidden_size, pooled_projection_dim=in_channels
        )
        self.proj_in = _aux_linear(in_channels, hidden_size, bias=True, model_config=model_config)
        self.token_refiner = HunyuanVideo15IndividualTokenRefiner(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_width_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            attention_bias=attention_bias,
            model_config=model_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled_projections = hidden_states.mean(dim=1)
        else:
            original_dtype = hidden_states.dtype
            mask_float = attention_mask.float().unsqueeze(-1)
            pooled_projections = (hidden_states * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            pooled_projections = pooled_projections.to(original_dtype)

        temb = self.time_text_embed(timestep, pooled_projections)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.token_refiner(hidden_states, temb, attention_mask)

        return hidden_states


class HunyuanVideo15ByT5TextProjection(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: int,
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        # Use eps=1e-5 for torch parity
        self.norm = LayerNorm(hidden_size=in_features, eps=1e-5)
        self.linear_1 = _aux_linear(in_features, hidden_size, model_config=model_config)
        self.linear_2 = _aux_linear(hidden_size, hidden_size, model_config=model_config)
        self.linear_3 = _aux_linear(hidden_size, out_features, model_config=model_config)
        self.act_fn = F.gelu

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(encoder_hidden_states)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_3(hidden_states)
        return hidden_states


class HunyuanVideo15Attention(Attention):
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
            num_attention_heads=num_attention_heads,
            head_dim=attention_head_dim,
            hidden_size=dim,
            config=config,
            qk_norm_mode="per_head",
            qkv_mode=QKVMode.FUSE_QKV,
            qk_norm=True,
        )

        self.heads = num_attention_heads
        self.head_dim = attention_head_dim

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

        self.to_out = torch.nn.ModuleList(
            [
                Linear(
                    dim,
                    dim,
                    bias=True,
                    dtype=dtype,
                    mapping=self.mapping,
                    quant_config=self.quant_config,
                    skip_create_weights_in_init=self.skip_create_weights_in_init,
                    force_dynamic_quantization=self.force_dynamic_quantization,
                ),
                torch.nn.Dropout(0.0),
            ]
        )
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
    def _head_norm(norm: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        # flashinfer RMSNorm only supports 2D/3D inputs; collapse the batch and
        # sequence dims so a 4D (B, S, H, D) tensor becomes 3D (B*S, H, D) before
        # the per-head norm over the head dim, then restore the original shape.
        b, s, h, d = x.shape
        return norm(x.reshape(b * s, h, d)).reshape(b, s, h, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # QKV Proj
        query, key, value = self.get_qkv(hidden_states)
        query = query.unflatten(2, (self.heads, -1))
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))

        # QK Normalization
        if self.qk_norm:
            query = self._head_norm(self.norm_q, query)
            key = self._head_norm(self.norm_k, key)

        # Rotary Embedding Application
        if image_rotary_emb is not None:
            query = _apply_rotary_emb(query, image_rotary_emb)
            key = _apply_rotary_emb(key, image_rotary_emb)

        # Encoder condition QKV projection and normalization
        if encoder_hidden_states is not None:
            encoder_query = self.add_q_proj(encoder_hidden_states)
            encoder_key = self.add_k_proj(encoder_hidden_states)
            encoder_value = self.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (self.heads, -1))
            encoder_key = encoder_key.unflatten(2, (self.heads, -1))
            encoder_value = encoder_value.unflatten(2, (self.heads, -1))

            encoder_query = self._head_norm(self.norm_added_q, encoder_query)
            encoder_key = self._head_norm(self.norm_added_k, encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        batch_size, seq_len, *_ = query.shape
        key_padding_mask = torch.nn.functional.pad(
            attention_mask, (seq_len - attention_mask.shape[1], 0), value=True
        ).bool()

        hidden_states = self._attn_impl(query, key, value, key_padding_mask=key_padding_mask)

        # hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            hidden_states = self.to_out[0](hidden_states)
            hidden_states = self.to_out[1](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanVideo15TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
        config: Optional[DiffusionModelConfig] = None,
        layer_idx=0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = HunyuanVideo15AdaLayerNormZero(hidden_size, model_config=config)
        self.norm1_context = HunyuanVideo15AdaLayerNormZero(hidden_size, model_config=config)

        self.attn = HunyuanVideo15Attention(
            dim, num_attention_heads, attention_head_dim, eps, dtype, config, layer_idx
        )

        self.norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = HunyuanVideo15FeedForward(
            hidden_size,
            mult=mlp_ratio,
            activation_fn="gelu-approximate",
            dtype=dtype,
            model_config=config,
            layer_idx=layer_idx,
        )

        self.norm2_context = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = HunyuanVideo15FeedForward(
            hidden_size,
            mult=mlp_ratio,
            activation_fn="gelu-approximate",
            dtype=dtype,
            model_config=config,
            layer_idx=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(
            1
        )

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideo15Transformer3DModel(BaseDiffusionModel):
    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__(model_config)

        vgm = model_config.visual_gen_mapping
        num_heads = getattr(model_config.pretrained_config, "num_attention_heads", 48)
        self.sharder = SequenceSharder.from_vgm(vgm, num_attention_heads=num_heads)

        pretrained_config = model_config.pretrained_config

        dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        skip_create_weights = model_config.skip_create_weights_in_init
        force_dynamic_quant = model_config.force_dynamic_quantization

        # Extract parameters from pretrained config
        # Defaults are derived from the '720p_t2v` variant`
        attn_mode = getattr(pretrained_config, "attn_mode", "flash")
        attn_param = getattr(pretrained_config, "attn_param", None)
        concat_condition = getattr(pretrained_config, "concat_condition", True)
        glyph_byT5_v2 = getattr(pretrained_config, "glyph_byT5_v2", True)
        guidance_embed = getattr(pretrained_config, "guidance_embed", False)
        heads_num = getattr(pretrained_config, "heads_num", 16)
        hidden_size = getattr(pretrained_config, "hidden_size", 2048)
        ideal_resolution = getattr(pretrained_config, "ideal_resolution", "720p")
        ideal_task = getattr(pretrained_config, "ideal_task", "t2v")
        in_channels = getattr(pretrained_config, "in_channels", 32)
        is_reshape_temporal_channels = getattr(
            pretrained_config, "is_reshape_temporal_channels", False
        )
        mlp_act_type = getattr(pretrained_config, "mlp_act_type", "gelu_tanh")
        mlp_width_ratio = getattr(pretrained_config, "mlp_width_ratio", 4)
        mm_double_blocks_depth = getattr(pretrained_config, "mm_double_blocks_depth", 54)
        mm_single_blocks_depth = getattr(pretrained_config, "mm_single_blocks_depth", 0)
        out_channels = getattr(pretrained_config, "out_channels", 32)
        # Patch size differs from HF 1 vs (1, 1, 1)
        patch_size = getattr(pretrained_config, "patch_size", 1)
        qk_norm = getattr(pretrained_config, "qk_norm", True)
        qk_norm_type = getattr(pretrained_config, "qk_norm_type", "rms")
        qkv_bias = getattr(pretrained_config, "qkv_bias", True)
        rope_dim_list = getattr(pretrained_config, "rope_dim_list", [16, 56, 56])
        rope_theta = getattr(pretrained_config, "rope_theta", 256)
        text_pool_type = getattr(pretrained_config, "text_pool_type", None)
        text_projection = getattr(pretrained_config, "text_projection", "single_refiner")
        text_states_dim = getattr(pretrained_config, "text_states_dim", 3584)
        text_states_dim_2 = getattr(pretrained_config, "text_states_dim_2", None)
        use_attention_mask = getattr(pretrained_config, "use_attention_mask", True)
        use_cond_type_embedding = getattr(pretrained_config, "use_cond_type_embedding", True)
        use_meanflow = getattr(pretrained_config, "use_meanflow", False)
        vision_projection = getattr(pretrained_config, "vision_projection", "linear")
        vision_states_dim = getattr(pretrained_config, "vision_states_dim", 1152)

        num_attention_heads = getattr(pretrained_config, "num_attention_heads", 16)
        attention_head_dim = getattr(pretrained_config, "attention_head_dim", 128)
        image_embed_dim = getattr(pretrained_config, "image_embed_dim", 1152)
        num_refiner_layers = getattr(pretrained_config, "num_refiner_layers", 2)
        patch_size_t = getattr(pretrained_config, "patch_size_t", 1)
        rope_axes_dim = getattr(pretrained_config, "rope_axes_dim", (16, 56, 56))
        text_embed_dim = getattr(pretrained_config, "text_embed_dim", 3584)
        text_embed_2_dim = getattr(pretrained_config, "text_embed_2_dim", 1472)
        num_layers = getattr(pretrained_config, "num_layers", 54)
        mlp_ratio = getattr(pretrained_config, "mlp_ratio", 4.0)
        target_size = getattr(pretrained_config, "target_size", 640)

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        self.gradient_checkpointing = False

        # Store config for compatibility
        self.config = type(
            "Config",
            (),
            {
                "attn_mode": attn_mode,
                "attn_param": attn_param,
                "concat_condition": concat_condition,
                "glyph_byT5_v2": glyph_byT5_v2,
                "guidance_embed": guidance_embed,
                "heads_num": heads_num,
                "hidden_size": hidden_size,
                "ideal_resolution": ideal_resolution,
                "ideal_task": ideal_task,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "patch_size": patch_size,
                "inner_dim": inner_dim,
                "is_reshape_temporal_channels": is_reshape_temporal_channels,
                "mlp_act_type": mlp_act_type,
                "mlp_width_ratio": mlp_width_ratio,
                "mm_double_blocks_depth": mm_double_blocks_depth,
                "mm_single_blocks_depth": mm_single_blocks_depth,
                "qk_norm": qk_norm,
                "qk_norm_type": qk_norm_type,
                "qkv_bias": qkv_bias,
                "rope_dim_list": rope_dim_list,
                "rope_theta": rope_theta,
                "text_pool_type": text_pool_type,
                "text_projection": text_projection,
                "text_states_dim": text_states_dim,
                "text_states_dim_2": text_states_dim_2,
                "use_attention_mask": use_attention_mask,
                "use_cond_type_embedding": use_cond_type_embedding,
                "use_meanflow": use_meanflow,
                "vision_projection": vision_projection,
                "vision_states_dim": vision_states_dim,
                "patch_size_t": patch_size_t,
                "dtype": dtype,
                "target_size": target_size,
                "image_embed_dim": image_embed_dim,
                "force_dynamic_quantization": force_dynamic_quant,
                "skip_create_weights": skip_create_weights,
                "quant_config": quant_config,
            },
        )()

        self.x_embedder = HunyuanVideo15PatchEmbed(
            (patch_size_t, patch_size, patch_size), in_channels, inner_dim
        )

        self.image_embedder = HunyuanVideo15ImageProjection(
            image_embed_dim, inner_dim, model_config=model_config
        )
        self.context_embedder = HunyuanVideo15TokenRefiner(
            text_embed_dim,
            num_attention_heads,
            attention_head_dim,
            num_layers=num_refiner_layers,
            model_config=model_config,
        )
        self.context_embedder_2 = HunyuanVideo15ByT5TextProjection(
            text_embed_2_dim, 2048, inner_dim, model_config=model_config
        )

        self.time_embed = HunyuanVideo15TimeEmbedding(inner_dim, use_meanflow=use_meanflow)
        self.cond_type_embed = Embedding(3, inner_dim)

        self.rope = HunyuanVideo15RotaryPosEmbed(
            patch_size, patch_size_t, rope_axes_dim, rope_theta
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                HunyuanVideo15TransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    layer_idx=layer_idx,
                    config=model_config,
                    dtype=dtype,
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.norm_out = HunyuanVideo15AdaLayerNormContinuous(
            inner_dim, inner_dim, elementwise_affine=False, eps=1e-6, model_config=model_config
        )
        self.proj_out = _aux_linear(
            inner_dim,
            patch_size_t * patch_size * patch_size * out_channels,
            model_config=model_config,
        )

        self.apply_quant_config_exclude_modules()
        self.__post_init__()

    def apply_quant_config_exclude_modules(self) -> None:
        """Opt excluded Linears out of quantization (mirrors the Wan transformer)."""
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return
        no_quant_config = QuantConfig(kv_cache_quant_algo=quant_config.kv_cache_quant_algo)
        for name, module in self.named_modules():
            if (
                isinstance(module, Linear)
                and getattr(module, "quant_config", None) is not None
                and quant_config.is_module_excluded_from_quantization(name)
            ):
                module.quant_config = no_quant_config
                module._weights_created = False
                module.create_weights()

    def __post_init__(self) -> None:
        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep_r: Optional[torch.LongTensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        encoder_attention_mask_2: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size_t, self.config.patch_size, self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        timestep_r = timestep_r.to(self.config.dtype) if timestep_r else None
        temb = self.time_embed(timestep.to(self.config.dtype), timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)

        # qwen text embedding
        encoder_hidden_states = self.context_embedder(
            encoder_hidden_states, timestep, encoder_attention_mask
        )

        encoder_hidden_states_cond_emb = self.cond_type_embed(
            torch.zeros_like(encoder_hidden_states[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_cond_emb

        # byt5 text embedding
        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

        encoder_hidden_states_2_cond_emb = self.cond_type_embed(
            torch.ones_like(encoder_hidden_states_2[:, :, 0], dtype=torch.long)
        )
        encoder_hidden_states_2 = encoder_hidden_states_2 + encoder_hidden_states_2_cond_emb

        # image embed
        encoder_hidden_states_3 = self.image_embedder(image_embeds)
        is_t2v = torch.all(image_embeds == 0)
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = torch.zeros(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        else:
            encoder_attention_mask_3 = torch.ones(
                (batch_size, encoder_hidden_states_3.shape[1]),
                dtype=encoder_attention_mask.dtype,
                device=encoder_attention_mask.device,
            )
        encoder_hidden_states_3_cond_emb = self.cond_type_embed(
            2
            * torch.ones_like(
                encoder_hidden_states_3[:, :, 0],
                dtype=torch.long,
            )
        )
        encoder_hidden_states_3 = encoder_hidden_states_3 + encoder_hidden_states_3_cond_emb

        # reorder and combine text tokens: combine valid tokens first, then padding
        encoder_attention_mask = encoder_attention_mask.bool()
        encoder_attention_mask_2 = encoder_attention_mask_2.bool()
        encoder_attention_mask_3 = encoder_attention_mask_3.bool()
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2, image, image_mask in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        ):
            # Concatenate: [valid_image, valid_byt5, valid_mllm, invalid_image, invalid_byt5, invalid_mllm]
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        image[image_mask],  # valid image
                        text_2[text_mask_2],  # valid byt5
                        text[text_mask],  # valid mllm
                        image[~image_mask],  # invalid image
                        torch.zeros_like(text_2[~text_mask_2]),  # invalid byt5 (zeroed)
                        torch.zeros_like(text[~text_mask]),  # invalid mllm (zeroed)
                    ],
                    dim=0,
                )
            )

            # Apply same reordering to attention masks
            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        image_mask[image_mask],
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        image_mask[~image_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    encoder_attention_mask,
                    image_rotary_emb,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    encoder_attention_mask,
                    image_rotary_emb,
                )

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            -1,
            p_t,
            p_h,
            p_w,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)

    def load_weights(self, weights: dict) -> None:
        """Load weights into the transformer.

        Args:
            weights: Dictionary of parameter name -> tensor
        """

        # Remap HF checkpoint keys to our module attribute names
        weights = _remap_checkpoint_keys(weights)

        # Map fused QKV layer names to original HF checkpoint names
        # HF checkpoint has separate to_q, to_k, to_v / add_q_proj, add_k_proj, add_v_proj
        # We fuse them into qkv_proj / add_qkv_proj for better performance
        params_map = {
            "add_qkv_proj": ["add_q_proj", "add_k_proj", "add_v_proj"],
            "qkv_proj": ["to_q", "to_k", "to_v"],
        }

        loader = DynamicLinearWeightLoader(self.model_config, params_map=params_map)

        # Track prefixes of wrapper projectors whose sub-Linears are loaded
        # by the parent's load_weights — the generic Linear loader must skip
        # them (their FUSED weight modes would look for nonexistent checkpoint
        # keys via params_map and error).
        managed_prefixes = set()

        for name, module in tqdm(self.named_modules(), desc="Loading weights"):
            if any(name.startswith(p) for p in managed_prefixes):
                continue

            # Create weights for modules with skip_create_weights_in_init=True
            # This must be done before loading weights (following Wan pattern)
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

            if len(module._parameters) == 0:
                continue

            if isinstance(module, Embedding):
                # Embedding subclasses Linear but must never be weight-quantized
                weight_dicts = loader.get_linear_weights(module, name, weights)
                if weight_dicts:
                    module.load_weights(weight_dicts)
            elif isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)

                if weight_dicts:
                    loader.load_linear_weights(module, name, weight_dicts)
            else:
                module_weights = loader.filter_weights(name, weights)
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(
                            module_weights[param_name].to(self.model_config.torch_dtype)
                        )

    def post_load_weights(self) -> None:
        """Call post_load_weights on all Linear modules and convert embedders to target dtype."""
        # Convert time_text_embed components to target dtype
        target_dtype = self.model_config.torch_dtype
        if hasattr(self, "time_text_embed"):
            if hasattr(self.time_text_embed, "timestep_embedder"):
                self.time_text_embed.timestep_embedder.to(target_dtype)
            if hasattr(self.time_text_embed, "text_embedder"):
                self.time_text_embed.text_embedder.to(target_dtype)
            if hasattr(self.time_text_embed, "guidance_embedder"):
                self.time_text_embed.guidance_embedder.to(target_dtype)

        # Finalize per-module weights and normalize dtypes to the compute dtype.
        compute_dtype = self.config.dtype
        quantized_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        for _, module in self.named_modules():
            if isinstance(module, Linear):
                module.post_load_weights()
                # Non-quantized Linear weights (e.g. fp32 embedder projections) need the
                # compute dtype
                weight = getattr(module, "weight", None)
                if (
                    weight is not None
                    and weight.is_floating_point()
                    and weight.dtype not in quantized_dtypes
                ):
                    module.to(compute_dtype)
                continue
            for param in module._parameters.values():
                if param is not None and param.is_floating_point():
                    param.data = param.data.to(compute_dtype)
