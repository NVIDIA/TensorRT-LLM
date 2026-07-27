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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_glm_image import GlmImageRotaryPosEmbed
from tqdm import tqdm

from tensorrt_llm._torch.modules.embedding import Embedding
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.models.modeling import BaseDiffusionModel
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.quantization import DynamicLinearWeightLoader
from tensorrt_llm._torch.visual_gen.utils import SequenceSharder
from tensorrt_llm.models.modeling_utils import QuantConfig


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


class GlmImageGELU(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        self.proj = _aux_linear(dim_in, dim_out, bias=bias, model_config=model_config)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GlmImageLinearActivation(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        self.proj = _aux_linear(dim_in, dim_out, bias=bias, model_config=model_config)
        self.activation = F.silu

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return self.activation(hidden_states)


class GlmImageFeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        inner_dim=None,
        bias: bool = True,
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GlmImageGELU(dim, inner_dim, bias=bias, model_config=model_config)
        elif activation_fn == "gelu-approximate":
            act_fn = GlmImageGELU(
                dim, inner_dim, approximate="tanh", bias=bias, model_config=model_config
            )
        elif activation_fn == "linear-silu":
            act_fn = GlmImageLinearActivation(dim, inner_dim, bias=bias, model_config=model_config)
        else:
            raise ValueError(f"Unsupported activation_fn={activation_fn} for GlmImageFeedForward")

        self.net = torch.nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(torch.nn.Dropout(dropout))
        self.net.append(_aux_linear(inner_dim, dim_out, bias=bias, model_config=model_config))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class GlmImageTimestepEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: Optional[int] = None,
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        self.linear_1 = _aux_linear(in_channels, time_embed_dim, model_config=model_config)
        self.act = torch.nn.SiLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = _aux_linear(time_embed_dim, time_embed_dim_out, model_config=model_config)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class GlmImagePixArtAlphaTextProjection(torch.nn.Module):
    def __init__(
        self,
        in_features,
        hidden_size,
        out_features=None,
        act_fn="gelu_tanh",
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = _aux_linear(in_features, hidden_size, bias=True, model_config=model_config)
        if act_fn == "gelu_tanh":
            self.act_1 = torch.nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = torch.nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = _aux_linear(hidden_size, out_features, bias=True, model_config=model_config)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class GlmImageCombinedTimestepSizeEmbeddings(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        condition_dim: int,
        pooled_projection_dim: int,
        timesteps_dim: int = 256,
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=timesteps_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.condition_proj = Timesteps(
            num_channels=condition_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = GlmImageTimestepEmbedding(
            in_channels=timesteps_dim, time_embed_dim=embedding_dim, model_config=model_config
        )
        self.condition_embedder = GlmImagePixArtAlphaTextProjection(
            pooled_projection_dim, embedding_dim, act_fn="silu", model_config=model_config
        )

    def forward(
        self,
        timestep: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)

        crop_coords_proj = self.condition_proj(crop_coords.flatten()).view(crop_coords.size(0), -1)
        target_size_proj = self.condition_proj(target_size.flatten()).view(target_size.size(0), -1)

        # (B, 2 * condition_dim)
        condition_proj = torch.cat([crop_coords_proj, target_size_proj], dim=1)

        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        condition_emb = self.condition_embedder(condition_proj.to(dtype=hidden_dtype))

        conditioning = timesteps_emb + condition_emb
        conditioning = F.silu(conditioning)

        return conditioning


class GlmImageImageProjector(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        hidden_size: int = 2560,
        patch_size: int = 2,
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = _aux_linear(in_channels * patch_size**2, hidden_size, model_config=model_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channel, height, width = hidden_states.shape
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size

        hidden_states = hidden_states.reshape(
            batch_size,
            channel,
            post_patch_height,
            self.patch_size,
            post_patch_width,
            self.patch_size,
        )
        hidden_states = hidden_states.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        hidden_states = self.proj(hidden_states)

        return hidden_states


class GlmImageAdaLayerNormZero(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dim: int,
        model_config: Optional[DiffusionModelConfig] = None,
    ) -> None:
        super().__init__()

        self.norm = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_context = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.linear = _aux_linear(embedding_dim, 12 * dim, bias=True, model_config=model_config)

    def forward(
        self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype
        norm_hidden_states = self.norm(hidden_states).to(dtype=dtype)
        norm_encoder_hidden_states = self.norm_context(encoder_hidden_states).to(dtype=dtype)

        emb = self.linear(temb)
        (
            shift_msa,
            c_shift_msa,
            scale_msa,
            c_scale_msa,
            gate_msa,
            c_gate_msa,
            shift_mlp,
            c_shift_mlp,
            scale_mlp,
            c_scale_mlp,
            gate_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)

        hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        encoder_hidden_states = norm_encoder_hidden_states * (
            1 + c_scale_msa.unsqueeze(1)
        ) + c_shift_msa.unsqueeze(1)

        return (
            hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        )


class GlmImageAdaLayerNormContinuous(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        norm_type: str = "layer_norm",
        model_config: Optional[DiffusionModelConfig] = None,
    ):
        super().__init__()
        self.linear = _aux_linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias, model_config=model_config
        )
        if norm_type == "layer_norm":
            self.norm = torch.nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(hidden_size=embedding_dim, eps=eps, has_weights=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        emb = self.linear(conditioning_embedding.to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class GlmImageAttention(Attention):
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        batch_size, text_seq_length, _ = encoder_hidden_states.shape
        batch_size, image_seq_length, _ = hidden_states.shape
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # QKV Proj
        query, key, value = self.get_qkv(hidden_states)
        query = query.unflatten(2, (self.heads, -1))
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))

        # 2. QK normalization
        if self.qk_norm:
            query, key = self.apply_qk_norm(query, key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, text_seq_length:, :, :] = apply_rotary_emb(
                query[:, text_seq_length:, :, :],
                image_rotary_emb,
                sequence_dim=1,
                use_real_unbind_dim=-2,
            )
            key[:, text_seq_length:, :, :] = apply_rotary_emb(
                key[:, text_seq_length:, :, :],
                image_rotary_emb,
                sequence_dim=1,
                use_real_unbind_dim=-2,
            )

        # 4. Attention
        if attention_mask is not None:
            text_attn_mask = attention_mask
            assert text_attn_mask.dim() == 2, (
                "the shape of text_attn_mask should be (batch_size, text_seq_length)"
            )
            text_attn_mask = text_attn_mask.float().to(query.device)
            mix_attn_mask = torch.ones(
                (batch_size, text_seq_length + image_seq_length), device=query.device
            )
            mix_attn_mask[:, :text_seq_length] = text_attn_mask
            mix_attn_mask = mix_attn_mask.unsqueeze(2)
            attn_mask_matrix = mix_attn_mask @ mix_attn_mask.transpose(1, 2)
            attention_mask = (attn_mask_matrix > 0).unsqueeze(1).to(query.dtype)

        hidden_states = self._attn_impl(query, key, value, key_padding_mask=attention_mask)

        hidden_states = hidden_states.to(query.dtype)

        # 5. Output projection
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        return hidden_states, encoder_hidden_states


class GlmImageTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int = 2560,
        num_attention_heads: int = 64,
        attention_head_dim: int = 40,
        time_embed_dim: int = 512,
        eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
        config: Optional[DiffusionModelConfig] = None,
        layer_idx=0,
    ):
        super().__init__()

        # 1. Attention
        self.norm1 = GlmImageAdaLayerNormZero(time_embed_dim, dim, model_config=config)
        self.attn1 = GlmImageAttention(
            dim, num_attention_heads, attention_head_dim, eps, dtype, config, layer_idx
        )

        # 2. Feedforward
        self.norm2 = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm2_context = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.ff = GlmImageFeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate", model_config=config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[
            Union[
                Tuple[torch.Tensor, torch.Tensor],
                List[Tuple[torch.Tensor, torch.Tensor]],
            ]
        ] = None,
        attention_mask: Optional[Dict[str, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Timestep conditioning
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, temb)

        # 2. Attention
        attention_kwargs = attention_kwargs or {}

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            **attention_kwargs,
        )
        hidden_states = hidden_states + attn_hidden_states * gate_msa.unsqueeze(1)
        encoder_hidden_states = (
            encoder_hidden_states + attn_encoder_hidden_states * c_gate_msa.unsqueeze(1)
        )

        # 3. Feedforward
        norm_hidden_states = self.norm2(hidden_states) * (
            1 + scale_mlp.unsqueeze(1)
        ) + shift_mlp.unsqueeze(1)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (
            1 + c_scale_mlp.unsqueeze(1)
        ) + c_shift_mlp.unsqueeze(1)

        ff_output = self.ff(norm_hidden_states)
        ff_output_context = self.ff(norm_encoder_hidden_states)
        hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)

        return hidden_states, encoder_hidden_states


class GlmImageTransformer2DModel(BaseDiffusionModel):
    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__(model_config)

        vgm = model_config.visual_gen_mapping
        num_heads = getattr(model_config.pretrained_config, "num_attention_heads", 32)
        self.sharder = SequenceSharder.from_vgm(vgm, num_attention_heads=num_heads)

        pretrained_config = model_config.pretrained_config

        dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        skip_create_weights = model_config.skip_create_weights_in_init
        force_dynamic_quant = model_config.force_dynamic_quantization

        attention_head_dim = getattr(pretrained_config, "attention_head_dim", 128)
        condition_dim = getattr(pretrained_config, "condition_dim", 256)
        in_channels = getattr(pretrained_config, "in_channels", 16)
        num_attention_heads = getattr(pretrained_config, "num_attention_heads", 32)
        num_layers = getattr(pretrained_config, "num_layers", 30)
        out_channels = getattr(pretrained_config, "out_channels", 16)
        patch_size = getattr(pretrained_config, "patch_size", 2)
        prior_vq_quantizer_codebook_size = getattr(
            pretrained_config, "prior_vq_quantizer_codebook_size", 16384
        )
        text_embed_dim = getattr(pretrained_config, "text_embed_dim", 1472)
        time_embed_dim = getattr(pretrained_config, "time_embed_dim", 512)
        num_train_timesteps = getattr(pretrained_config, "num_train_timesteps", 1000)
        pooled_projection_dim = 2 * 2 * condition_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.config = type(
            "Config",
            (),
            {
                "attention_head_dim": attention_head_dim,
                "condition_dim": condition_dim,
                "in_channels": in_channels,
                "num_attention_heads": num_attention_heads,
                "num_layers": num_layers,
                "out_channels": out_channels,
                "patch_size": patch_size,
                "prior_vq_quantizer_codebook_size": prior_vq_quantizer_codebook_size,
                "text_embed_dim": text_embed_dim,
                "time_embed_dim": time_embed_dim,
                "num_train_timesteps": num_train_timesteps,
                "pooled_projection_dim": pooled_projection_dim,
                "inner_dim": inner_dim,
                "dtype": dtype,
                "quant_config": quant_config,
                "skip_create_weights": skip_create_weights,
                "force_dynamic_quant": force_dynamic_quant,
            },
        )

        # 1. RoPE
        self.rope = GlmImageRotaryPosEmbed(attention_head_dim, patch_size, theta=10000.0)

        # 2. Patch & Text-timestep embedding
        self.image_projector = GlmImageImageProjector(
            in_channels, inner_dim, patch_size, model_config=model_config
        )
        self.glyph_projector = GlmImageFeedForward(
            text_embed_dim,
            inner_dim,
            inner_dim=inner_dim,
            activation_fn="gelu",
            model_config=model_config,
        )
        self.prior_token_embedding = Embedding(prior_vq_quantizer_codebook_size, inner_dim)
        self.prior_projector = GlmImageFeedForward(
            inner_dim,
            inner_dim,
            inner_dim=inner_dim,
            activation_fn="linear-silu",
            model_config=model_config,
        )

        self.time_condition_embed = GlmImageCombinedTimestepSizeEmbeddings(
            embedding_dim=time_embed_dim,
            condition_dim=condition_dim,
            pooled_projection_dim=pooled_projection_dim,
            timesteps_dim=time_embed_dim,
            model_config=model_config,
        )

        # 3. Transformer blocks
        self.transformer_blocks = torch.nn.ModuleList(
            [
                GlmImageTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    time_embed_dim,
                    config=model_config,
                    dtype=dtype,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )

        # 4. Output projection
        self.norm_out = GlmImageAdaLayerNormContinuous(
            inner_dim, time_embed_dim, elementwise_affine=False, model_config=model_config
        )
        self.proj_out = _aux_linear(
            inner_dim, patch_size * patch_size * out_channels, model_config=model_config
        )

        self.gradient_checkpointing = False

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
        encoder_hidden_states: torch.Tensor,
        prior_token_id: torch.Tensor,
        prior_token_drop: torch.Tensor,
        timestep: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[
            Union[
                Tuple[torch.Tensor, torch.Tensor],
                List[Tuple[torch.Tensor, torch.Tensor]],
            ]
        ] = None,
    ) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
        """
        Args:
            timestep: Normalized scheduler timestep tensor in [0, 1].
        """
        batch_size, num_channels, height, width = hidden_states.shape

        # 1. RoPE
        if image_rotary_emb is None:
            image_rotary_emb = self.rope(hidden_states)

        # 2. Patch & Timestep embeddings
        p = self.config.patch_size
        post_patch_height = height // p
        post_patch_width = width // p

        hidden_states = self.image_projector(hidden_states)
        encoder_hidden_states = self.glyph_projector(encoder_hidden_states)
        prior_embedding = self.prior_token_embedding(prior_token_id)
        prior_embedding[prior_token_drop] *= 0.0
        prior_hidden_states = self.prior_projector(prior_embedding)

        hidden_states = hidden_states + prior_hidden_states

        # GlmImage timestep embeddings use the scheduler's 1000-step scale internally.
        timestep = timestep * self.config.num_train_timesteps
        temb = self.time_condition_embed(timestep, target_size, crop_coords, hidden_states.dtype)

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    attention_kwargs,
                )

        # 4. Output norm & projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, -1, p, p
        )

        # Rearrange tensor from (B, H_p, W_p, C, p, p) to (B, C, H_p * p, W_p * p)
        output = hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: dict) -> None:
        """Load weights into the transformer.

        Args:
            weights: Dictionary of parameter name -> tensor
        """

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
        """Call post_load_weights on all Linear modules and normalize dtypes."""
        compute_dtype = self.model_config.torch_dtype
        quantized_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        for _, module in self.named_modules():
            if isinstance(module, Linear):
                module.post_load_weights()
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
