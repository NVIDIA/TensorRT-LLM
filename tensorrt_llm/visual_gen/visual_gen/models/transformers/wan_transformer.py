# Adapted from: https://github.com/huggingface/diffusers/blob/9d313fc718c8ace9a35f07dad9d5ce8018f8d216/src/diffusers/models/transformers/transformer_wan.py
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanTransformer3DModel,
    _get_added_kv_projections,
    _get_qkv_projections,
)
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from torch import nn

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.attention import ditAttnProcessor
from visual_gen.layers.linear import ditLinear
from visual_gen.models.transformers.base_transformer import ditBaseTransformer
from visual_gen.utils import dit_sp_gather, dit_sp_split
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


class ditWanAttnProcessor(ditAttnProcessor):
    def __init__(self):
        logger.debug("Initializing ditWanAttnProcessor")
        super().__init__()

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # Leverage ditAttnProcessor's attn_impl
        hidden_states = self.visual_gen_attn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            tensor_layout="NHD",
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class ditWanTransformer3DModel(WanTransformer3DModel, ditBaseTransformer):
    """Implementation of WanTransformer3DModel."""

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super(ditWanTransformer3DModel, self).__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self._post_init()

    def _post_init(self):
        for name, module in self.named_modules():
            # only replace self-attention layers
            # the cross-attention has very small kv, thus doesn't need cp and we don't split its kv
            if isinstance(module, WanAttention) and "attn1" in name:
                attn_processor = ditWanAttnProcessor()
                attn_processor.name = name
                module.set_processor(attn_processor)

        # Collect all linear modules to replace first to avoid modifying during iteration
        linear_modules_to_replace = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                linear_modules_to_replace.append((name, module))

        # Replace linear modules
        for name, module in linear_modules_to_replace:
            if "block" not in name:
                # Only quantize the linear layers in the transformer blocks to keep accuracy
                continue
            linear = ditLinear.from_linear(module)
            linear.name = name
            # Use proper nested attribute setting for complex names like "transformer_blocks.0.attn.to_q"
            parent = self
            attrs = name.split(".")
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, attrs[-1], linear)

    def _teacache_modulated_inp(
        self, timestep_proj: torch.Tensor, temb: torch.Tensor
    ) -> torch.Tensor:
        if not TeaCacheConfig.enable_teacache():
            return None

        e = temb.to(torch.float32)
        e0 = timestep_proj.to(torch.float32)
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        modulated_inp = e0 if TeaCacheConfig.use_ret_steps() else e
        return modulated_inp

    def run_transformer_blocks(
        self, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
    ):
        # Do not split encoder hidden states here. For some models, encoder hidden states are used in cross-attention, since they are very short, no need to split it.
        hidden_states = dit_sp_split(hidden_states, dim=1)
        if isinstance(rotary_emb, (list, tuple)):
            rotary_emb = [dit_sp_split(emb, dim=1) for emb in rotary_emb]
        else:
            rotary_emb = dit_sp_split(rotary_emb, dim=1)

        for idx, block in enumerate(self.blocks):
            PipelineConfig.set_config(current_dit_block_id=idx)
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        hidden_states = dit_sp_gather(hidden_states)

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep,
                encoder_hidden_states,
                encoder_hidden_states_image,
                timestep_seq_len=ts_seq_len,
            )
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # 4. Transformer blocks
        if TeaCacheConfig.enable_teacache():
            modulated_inp = self._teacache_modulated_inp(timestep_proj, temb)
            should_calc, hidden_states = self._calc_teacache_distance(modulated_inp, hidden_states)
            if should_calc:
                original_hidden_states = hidden_states.clone()
                hidden_states = self.run_transformer_blocks(
                    hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
                self._update_teacache_residual(original_hidden_states, hidden_states)
        else:
            hidden_states = self.run_transformer_blocks(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
            )

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
            hidden_states
        )
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
