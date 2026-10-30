# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VDN transformer using the existing packed MiniMax-H3 model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig

from .transformer_minimax_h3 import (
    MiniMaxH3StaticContext,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
    MiniMaxH3TransformerOutput,
)
from .vdn_attention import VDNHybridAttention, VDNLayout


def _modulate(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    shift: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    normed = F.rms_norm(hidden, (hidden.shape[-1],), weight, eps)
    return normed * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def _residual(
    hidden: torch.Tensor, output: torch.Tensor, gate: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    return hidden + gate.index_select(0, indices) * output


_compiled_modulate = torch.compile(_modulate, dynamic=False)
_compiled_residual = torch.compile(_residual, dynamic=False)


class VDNMiniMaxH3TransformerBlock(MiniMaxH3TransformerBlock):
    """Reuse every H3 block component with VDN's fused FP32 pointwise arithmetic."""

    def __init__(self, block: MiniMaxH3TransformerBlock) -> None:
        torch.nn.Module.__init__(self)
        self.norm1, self.norm2 = block.norm1, block.norm2
        self.attn, self.ff, self.adaln_proj = block.attn, block.ff, block.adaln_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        key_padding_mask: torch.Tensor | None = None,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not hidden_states.is_cuda:
            return super().forward(
                hidden_states, temb, adaln_indices, rotary_emb, key_padding_mask, timestep
            )
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = self.adaln_proj(temb)
        normalized = _compiled_modulate(
            hidden_states,
            self.norm1.weight,
            self.norm1.variance_epsilon,
            scale_a,
            shift_a,
            adaln_indices,
        )
        attended = self.attn(normalized, rotary_emb, key_padding_mask, timestep)
        hidden_states = _compiled_residual(hidden_states, attended, gate_a, adaln_indices)
        normalized = _compiled_modulate(
            hidden_states,
            self.norm2.weight,
            self.norm2.variance_epsilon,
            scale_f,
            shift_f,
            adaln_indices,
        )
        output = self.ff(normalized.reshape(-1, normalized.shape[-1])).reshape_as(normalized)
        return _compiled_residual(hidden_states, output, gate_f, adaln_indices)


@dataclass
class VDNStaticContext(MiniMaxH3StaticContext):
    position_ids: torch.Tensor
    layout: VDNLayout | None = None


class VDNMiniMaxH3Transformer3DModel(MiniMaxH3Transformer3DModel):
    """H3 backbone with checkpoint-compatible VDN attention modules."""

    def __init__(self, model_config: DiffusionModelConfig) -> None:
        if model_config.mapping.world_size != 1:
            raise NotImplementedError("VDN-H3 currently supports single-GPU execution.")
        if model_config.attention.backend not in ("VANILLA", "FA4"):
            raise NotImplementedError("VDN-H3 window attention requires VANILLA or FA4.")
        if model_config.quant_config and model_config.quant_config.quant_algo:
            raise NotImplementedError("VDN-H3 currently supports BF16 weights only.")
        super().__init__(model_config)
        if self.sharder.size != 1:
            raise NotImplementedError("VDN-H3 sequence parallelism is not implemented.")
        for index, block in enumerate(self.transformer_blocks):
            block.attn = VDNHybridAttention(
                block.attn,
                self.config.hidden_size,
                model_config.pretrained_config.vdn_transform,
                dtype=model_config.torch_dtype,
            )
            self.transformer_blocks[index] = VDNMiniMaxH3TransformerBlock(block)

    def prepare_static_context(
        self, encoder_hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> VDNStaticContext:
        context = super().prepare_static_context(encoder_hidden_states, position_ids)
        return VDNStaticContext(context.text_embeds, context.rotary_emb, position_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep: torch.Tensor,
        conditioning_timesteps: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor | None,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict[str, object] | None = None,
        return_dict: bool = True,
        static_context: VDNStaticContext | None = None,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        if static_context is None:
            if position_ids is None or encoder_hidden_states is None:
                raise ValueError("VDN requires positions and text or a VDN static context.")
            static_context = self.prepare_static_context(encoder_hidden_states, position_ids)
        if static_context.layout is None:
            static_context.layout = VDNLayout.from_indices(
                static_context.position_ids, video_indices, text_indices
            )
        for block in self.transformer_blocks:
            block.attn.layout = static_context.layout
        return super().forward(
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            conditioning_timesteps,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
            attention_kwargs,
            return_dict,
            static_context,
        )

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        # Only trunk attention is wrapped. Refiner attention remains dense.
        parameters = dict(self.named_parameters())
        remapped = {}
        for name, weight in weights.items():
            if name.startswith("transformer_blocks.") and ".attn." in name:
                tail = name.split(".attn.", 1)[1]
                if tail.startswith(("to_q.", "to_k.", "to_v.", "to_out.", "norm_q.", "norm_k.")):
                    name = name.replace(".attn.", ".attn.orig.", 1)
            if any(
                part in name for part in (".linear_attention.", ".softmax_gate.", ".to_out_linear.")
            ):
                if name not in parameters or parameters[name].shape != weight.shape:
                    raise ValueError(f"Unknown or incompatible VDN branch tensor: {name}")
            remapped[name] = weight
        super().load_weights(remapped)
