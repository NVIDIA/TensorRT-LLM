# Adapted from: https://github.com/huggingface/diffusers/blob/9d313fc718c8ace9a35f07dad9d5ce8018f8d216/src/diffusers/models/transformers/transformer_flux.py
# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxAttnProcessor,
    FluxIPAdapterAttnProcessor,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    _get_qkv_projections,
)
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch import nn

from visual_gen.configs.diffusion_cache import TeaCacheConfig
from visual_gen.configs.op_manager import LinearOpManager
from visual_gen.configs.parallel import DiTParallelConfig
from visual_gen.configs.pipeline import PipelineConfig
from visual_gen.layers.attention import ditAttnProcessor
from visual_gen.layers.linear import ditLinear
from visual_gen.models.transformers.base_transformer import ditBaseTransformer
from visual_gen.models.utils import disable_weight_management
from visual_gen.utils import dit_sp_gather, dit_sp_split
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


class ditFluxAttnProcessor(ditAttnProcessor):
    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = self.visual_gen_attn(
            query, key, value, attn_mask=attention_mask, tensor_layout="NHD"
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class ditFluxIPAdapterAttnProcessor(FluxIPAdapterAttnProcessor, ditAttnProcessor):
    """Flux Attention processor for IP-Adapter."""

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens=(4,),
        scale=1.0,
        device=None,
        dtype=None,
    ):
        super(ditFluxIPAdapterAttnProcessor, self).__init__(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_tokens=num_tokens,
            scale=scale,
            device=device,
            dtype=dtype,
        )

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        ip_hidden_states: Optional[List[torch.Tensor]] = None,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)
        ip_query = query

        if encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

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
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            # IP-adapter
            ip_attn_output = torch.zeros_like(hidden_states)

            for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
                ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
            ):
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = ip_key.view(batch_size, -1, attn.heads, attn.head_dim)
                ip_value = ip_value.view(batch_size, -1, attn.heads, attn.head_dim)

                (ip_query, ip_key, ip_value) = (
                    x.permute(0, 2, 1, 3) for x in (ip_query, ip_key, ip_value)
                )
                current_ip_hidden_states = F.scaled_dot_product_attention(
                    ip_query,
                    ip_key,
                    ip_value,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=False,
                )
                current_ip_hidden_states = current_ip_hidden_states.permute(0, 2, 1, 3)

                current_ip_hidden_states = current_ip_hidden_states.reshape(
                    batch_size, -1, attn.heads * attn.head_dim
                )
                current_ip_hidden_states = current_ip_hidden_states.to(ip_query.dtype)
                ip_attn_output += scale * current_ip_hidden_states

            return hidden_states, encoder_hidden_states, ip_attn_output
        else:
            return hidden_states


@maybe_allow_in_graph
class ditSvdFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0
    ):
        super(ditSvdFluxSingleTransformerBlock, self).__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            mlp_ratio=mlp_ratio,
        )
        self.proj_out0 = nn.Linear(dim, dim)
        self.proj_out1 = nn.Linear(self.mlp_hidden_dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states0 = self.proj_out0(attn_output)
        hidden_states1 = self.proj_out1(mlp_hidden_states)
        gate = gate.unsqueeze(1)

        hidden_states = gate * (hidden_states0 + hidden_states1)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :text_seq_len],
            hidden_states[:, text_seq_len:],
        )
        return encoder_hidden_states, hidden_states


class ditFluxTransformer2DModel(FluxTransformer2DModel, ditBaseTransformer):
    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super(ditFluxTransformer2DModel, self).__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

        if LinearOpManager.linear_type == "svd-nvfp4":
            self.single_transformer_blocks = nn.ModuleList(
                [
                    ditSvdFluxSingleTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    )
                    for _ in range(num_single_layers)
                ]
            )
        self._post_init()

    def _post_init(self):
        for name, module in self.named_modules():
            if isinstance(module, FluxAttention):
                origin_attn_processor = module.get_processor()
                if isinstance(origin_attn_processor, FluxAttnProcessor):
                    attn_processor = ditFluxAttnProcessor()
                    attn_processor.name = name
                    module.set_processor(attn_processor)
                elif isinstance(origin_attn_processor, FluxIPAdapterAttnProcessor):
                    attn_processor = ditFluxIPAdapterAttnProcessor(
                        hidden_size=origin_attn_processor.hidden_size,
                        cross_attention_dim=origin_attn_processor.cross_attention_dim,
                    )
                    attn_processor.scale = origin_attn_processor.scale
                    attn_processor.to_k_ip = origin_attn_processor.to_k_ip
                    attn_processor.to_v_ip = origin_attn_processor.to_v_ip
                    attn_processor.name = name
                    module.set_processor(attn_processor)
                else:
                    logger.warning(
                        f"Unsupported attn processor: {type(origin_attn_processor)} for {name}"
                    )

        # Collect all linear modules to replace first to avoid modifying during iteration
        linear_modules_to_replace = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                linear_modules_to_replace.append((name, module))

        # Replace linear modules
        for name, module in linear_modules_to_replace:
            linear = ditLinear.from_linear(module)
            linear.name = name
            # Use proper nested attribute setting for complex names like "transformer_blocks.0.attn.to_q"
            parent = self
            attrs = name.split(".")
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, attrs[-1], linear)

    def _calculate_blocks(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_blocks_repeat,
        controlnet_single_block_samples,
    ):
        global_block_index = 0
        for index_block, block in enumerate(self.transformer_blocks):
            PipelineConfig.set_config(current_dit_block_id=global_block_index)
            global_block_index += 1
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block // interval_control]
                    )

        for index_block, block in enumerate(self.single_transformer_blocks):
            PipelineConfig.set_config(current_dit_block_id=global_block_index)
            global_block_index += 1
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states = (
                    hidden_states + controlnet_single_block_samples[index_block // interval_control]
                )

        return hidden_states

    def _sp_split(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        sp_size = DiTParallelConfig.sp_size()
        if sp_size > 1:
            assert hidden_states.shape[1] % sp_size == 0, (
                f"Hidden states({hidden_states.shape}) sequence length must be divisible by sp_size({sp_size})"
            )
            assert encoder_hidden_states.shape[1] % sp_size == 0, (
                f"Encoder hidden states({encoder_hidden_states.shape}) "
                f"sequence length must be divisible by sp_size({sp_size})"
            )
            assert image_rotary_emb[0].shape[0] % sp_size == 0, (
                f"Image rotary emb({image_rotary_emb[0].shape}) sequence length must be divisible by sp_size({sp_size})"
            )

            txt_seq_len = encoder_hidden_states.shape[1]
            hidden_states = dit_sp_split(hidden_states, dim=1)
            encoder_hidden_states = dit_sp_split(encoder_hidden_states, dim=1)
            chunked_image_rotary_emb = []
            for emb in image_rotary_emb:
                txt_rotary_emb = dit_sp_split(emb[:txt_seq_len], dim=0)
                latent_rotary_emb = dit_sp_split(emb[txt_seq_len:], dim=0)
                chunked_image_rotary_emb.append(
                    torch.cat([txt_rotary_emb, latent_rotary_emb], dim=0)
                )
            image_rotary_emb = tuple(chunked_image_rotary_emb)

        return hidden_states, encoder_hidden_states, image_rotary_emb

    def run_transformer_blocks(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        image_rotary_emb,
        joint_attention_kwargs,
        controlnet_block_samples,
        controlnet_single_block_samples,
        controlnet_blocks_repeat,
    ):
        hidden_states, encoder_hidden_states, image_rotary_emb = self._sp_split(
            hidden_states, encoder_hidden_states, image_rotary_emb
        )

        hidden_states = self._calculate_blocks(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb,
            joint_attention_kwargs,
            controlnet_block_samples,
            controlnet_blocks_repeat,
            controlnet_single_block_samples,
        )

        hidden_states = dit_sp_gather(hidden_states)

        return hidden_states

    def run_pre_processing(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        guidance: torch.Tensor = None,
    ):
        """Pre-processing: embeddings and position encoding. Can be wrapped with CUDA Graph."""
        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        return hidden_states, encoder_hidden_states, temb, image_rotary_emb

    def run_post_processing(self, hidden_states: torch.Tensor, temb: torch.Tensor):
        """Post-processing: output norm and projection. Can be wrapped with CUDA Graph."""
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return output

    def run_teacache_check(self, hidden_states: torch.Tensor, temb: torch.Tensor):
        """TeaCache distance check. Can be wrapped with CUDA Graph (always runs)."""
        with disable_weight_management():
            modulated_inp, _, _, _, _ = self.transformer_blocks[0].norm1(hidden_states, emb=temb)
        return modulated_inp

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        # === Pre-processing (can be CUDA Graph wrapped) ===
        hidden_states, encoder_hidden_states, temb, image_rotary_emb = self.run_pre_processing(
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            img_ids,
            txt_ids,
            guidance,
        )

        if (
            joint_attention_kwargs is not None
            and "ip_adapter_image_embeds" in joint_attention_kwargs
        ):
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # === TeaCache logic with conditional branching ===
        if TeaCacheConfig.enable_teacache():
            # TeaCache check (can be CUDA Graph wrapped separately)
            modulated_inp = self.run_teacache_check(hidden_states, temb)
            should_calc, hidden_states = self._calc_teacache_distance(modulated_inp, hidden_states)
            if should_calc:
                original_hidden_states = hidden_states.clone()
                hidden_states = self.run_transformer_blocks(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                    controlnet_block_samples,
                    controlnet_single_block_samples,
                    controlnet_blocks_repeat,
                )
                self._update_teacache_residual(original_hidden_states, hidden_states)
        else:
            hidden_states = self.run_transformer_blocks(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
                controlnet_block_samples,
                controlnet_single_block_samples,
                controlnet_blocks_repeat,
            )

        # === Post-processing (can be CUDA Graph wrapped) ===
        output = self.run_post_processing(hidden_states, temb)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
