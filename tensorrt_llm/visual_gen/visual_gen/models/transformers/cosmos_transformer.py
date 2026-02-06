# Adapted from: https://github.com/huggingface/diffusers/blob/9d313fc718c8ace9a35f07dad9d5ce8018f8d216/src/diffusers/models/transformers/transformer_cosmos.py
# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, is_torchvision_available, scale_lora_layers, unscale_lora_layers
from diffusers.utils.logging import get_logger
from torch import nn

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.attention import ditAttnProcessor
from visual_gen.layers.linear import ditLinear
from visual_gen.models.transformers.base_transformer import ditBaseTransformer
from visual_gen.models.utils import disable_weight_management
from visual_gen.utils import dit_sp_gather, dit_sp_split

logger = get_logger(__name__)

if is_torchvision_available():
    from torchvision import transforms
else:
    transforms = None


def _set_module_by_qualified_name(root: nn.Module, qualified_name: str, new_module: nn.Module):
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


class ditCosmosAttnProcessor2_0(ditAttnProcessor):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. QKV projections
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        # 3. Apply RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        # 4. Prepare for GQA
        query_idx = query.size(3)
        key_idx = key.size(3)
        value_idx = value.size(3)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)

        # 5. Attention
        hidden_states = self.visual_gen_attn(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, tensor_layout="HND"
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).type_as(query)

        # 6. Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class ditCosmosTransformer3DModel(CosmosTransformer3DModel, ditBaseTransformer):
    """
    Refer to CosmosTransformer3DModel in diffusers
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
        extra_pos_embed_type: Optional[str] = "learnable",
        use_crossattn_projection: bool = False,
        crossattn_proj_in_channels: int = 1024,
        encoder_hidden_states_channels: int = 1024,
    ) -> None:
        super(ditCosmosTransformer3DModel, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            text_embed_dim=text_embed_dim,
            adaln_lora_dim=adaln_lora_dim,
            max_size=max_size,
            patch_size=patch_size,
            rope_scale=rope_scale,
            concat_padding_mask=concat_padding_mask,
            extra_pos_embed_type=extra_pos_embed_type,
            use_crossattn_projection=use_crossattn_projection,
            crossattn_proj_in_channels=crossattn_proj_in_channels,
            encoder_hidden_states_channels=encoder_hidden_states_channels,
        )
        self._post_init()

    def _post_init(self):
        self._replace_attn_processors()
        self._replace_linear_modules_meta_safe()

    def _replace_attn_processors(self):
        for name, module in self.named_modules():
            if isinstance(module, Attention) and "attn1" in name:
                attn_processor = ditCosmosAttnProcessor2_0()
                setattr(attn_processor, "name", name)
                module.set_processor(attn_processor)

    def _replace_linear_modules_meta_safe(self):
        linear_modules_to_replace = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                linear_modules_to_replace.append((name, module))

        for name, module in linear_modules_to_replace:
            visual_gen_linear = ditLinear.from_linear(module)
            setattr(visual_gen_linear, "name", name)
            _set_module_by_qualified_name(self, name, visual_gen_linear)

    def run_transformer_blocks(
        self,
        hidden_states,
        encoder_hidden_states,
        embedded_timestep,
        temb,
        image_rotary_emb,
        extra_pos_emb,
        attention_mask,
    ):
        hidden_states = dit_sp_split(hidden_states, dim=1)
        image_rotary_emb = [dit_sp_split(emb, dim=0) for emb in image_rotary_emb]
        if embedded_timestep.ndim == 3:
            embedded_timestep = dit_sp_split(embedded_timestep, dim=1)
        if temb.ndim == 3:
            temb = dit_sp_split(temb, dim=1)
        if extra_pos_emb is not None:
            extra_pos_emb = dit_sp_split(extra_pos_emb, dim=1)
        for idx, block in enumerate(self.transformer_blocks):
            PipelineConfig.set_config(current_dit_block_id=idx)
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                embedded_timestep=embedded_timestep,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                extra_pos_emb=extra_pos_emb,
                attention_mask=attention_mask,
            )
        hidden_states = dit_sp_gather(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        disallow_teacache: bool = False,
        **extra_transformer_kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, _, num_frames, height, width = hidden_states.shape

        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            if transforms is None:
                raise ImportError(
                    "concat_padding_mask=True need torchvision, please install torchvision or disable concat_padding_mask"
                )
            padding_mask_resized = transforms.functional.resize(
                padding_mask,
                list(hidden_states.shape[-2:]),
                interpolation=transforms.InterpolationMode.NEAREST,
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask_resized.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
            )

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None

        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states.flatten(1, 3)

        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (
                batch_size,
                1,
                num_frames,
                1,
                1,
            ), f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            flat_timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, flat_timestep)
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )
        else:
            raise AssertionError(f"Unsupported timestep ndim: {timestep.ndim}")

        if self.config.use_crossattn_projection:
            encoder_hidden_states = self.crossattn_proj(encoder_hidden_states)

        if TeaCacheConfig.enable_teacache() and not disallow_teacache:
            with disable_weight_management():
                if extra_pos_emb is not None:
                    modulated_inp, _ = self.transformer_blocks[0].norm1(
                        hidden_states + extra_pos_emb, embedded_timestep=embedded_timestep, temb=temb
                    )
                else:
                    modulated_inp, _ = self.transformer_blocks[0].norm1(
                        hidden_states, embedded_timestep=embedded_timestep, temb=temb
                    )
            should_calc, hidden_states = self._calc_teacache_distance(modulated_inp, hidden_states)
            if should_calc:
                original_hidden_states = hidden_states.clone()
                hidden_states = self.run_transformer_blocks(
                    hidden_states,
                    encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                )
                self._update_teacache_residual(original_hidden_states, hidden_states)
        else:
            hidden_states = self.run_transformer_blocks(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                embedded_timestep=embedded_timestep,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                extra_pos_emb=extra_pos_emb,
                attention_mask=attention_mask,
            )

        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
