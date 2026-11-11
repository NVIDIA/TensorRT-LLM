"""CombinedDiffusionNABlock: context_and_x inject + full-volume attn + residual MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.combined.attn import full as residual_attn
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.combined.attn import full_with_keyframes as residual_attn_with_keyframes
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.combined.context import combined as inject_context
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.combined.mlp import residual_mlp

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import KeyframePlanes


class CombinedDiffusionNABlock(DiffusionNABlock):
    """Combined-context diffusion block: ``forward`` / ``forward_combined``."""

    def forward_combined_with_keyframes(
        self,
        context_and_x: torch.Tensor,
        keyframe_context_and_x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        keyframe_planes: KeyframePlanes,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual-stream block; returns the two updated x halves (not the concat buffers).
        Each stream gets its own ``context_proj(context)`` injection from its own
        ``[context | x]`` buffer, then both meet in one joint attention softmax, then each
        runs the shared MLP. Invalid keyframe planes are re-zeroed on the way out -- unlike
        the deterministic ``NABlock``, which leaves that to the decoder's post-upsample
        masking. Both asymmetries are upstream's.
        """
        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        x = inject_context(context_and_x, self.context_proj.weight, self.context_proj.bias)
        keyframe_x = inject_context(keyframe_context_and_x, self.context_proj.weight, self.context_proj.bias)
        x, keyframe_x = residual_attn_with_keyframes(
            x,
            keyframe_x,
            self.attn,
            self.norm1,
            scale_msa,
            shift_msa,
            keyframe_planes,
        )
        x = residual_mlp(x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        keyframe_x = residual_mlp(keyframe_x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        return x, keyframe_x * keyframe_planes.valid.device_tensor[None, :, None, None, None]

    def forward_combined(
        self,
        context_and_x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        x = inject_context(context_and_x, self.context_proj.weight, self.context_proj.bias)
        x = residual_attn(x, self.attn, self.norm1, scale_msa, shift_msa)
        x = residual_mlp(x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        return x

    def forward(
        self,
        context_and_x: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.forward_combined(context_and_x, modulation)
