"""ChunkedDiffusionNABlock: deferred inject + W-chunked attn + modulating MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.chunked.attn import chunked as residual_attn
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.chunked.attn import (
    chunked_with_keyframes as residual_attn_with_keyframes,
)
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.chunked.context import deferred as inject_context
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.chunked.context import deferred_keyframes as inject_keyframe_context
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.chunked.mlp import residual_modulating_mlp
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.transformer.layers import LinearPixelShuffleUpsample

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.video_vae.keyframes import KeyframePlanes


class ChunkedDiffusionNABlock(DiffusionNABlock):
    """Deferred-context diffusion block.
    ``forward`` / ``forward_x_ctx``: in-place inject (eager) then attn+mlp.
    ``forward_attn_mlp``: opaque void-mutate residual + SwiGLU — this is what
    ChunkedCompile ``torch.compile``s (do not compile inject).
    ``stage4_upsample`` is wired by ``apply`` to ``decoder.upsamples[3]`` so
    inject runs sequential upsample Linear then ``context_proj``.
    """

    stage4_upsample: LinearPixelShuffleUpsample | None

    def forward_attn_mlp(
        self,
        x: torch.Tensor,
        scale_msa: torch.Tensor,
        shift_msa: torch.Tensor,
        scale_mlp: torch.Tensor,
        shift_mlp: torch.Tensor,
    ) -> torch.Tensor:
        """In-place chunked NA residual + modulating SwiGLU (compile target)."""
        x = residual_attn(x, self.attn, self.norm1, scale_msa, shift_msa)
        x = residual_modulating_mlp(x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        return x

    def forward_attn_mlp_with_keyframes(
        self,
        x: torch.Tensor,
        keyframe_x: torch.Tensor,
        scale_msa: torch.Tensor,
        shift_msa: torch.Tensor,
        scale_mlp: torch.Tensor,
        shift_mlp: torch.Tensor,
        keyframe_planes: KeyframePlanes,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual-stream chunked NA residual + modulating SwiGLU, both in place.
        Invalid keyframe planes are re-zeroed on the way out, matching
        :meth:`~ltx_core.model.video_vae.transformer.combined.block.CombinedDiffusionNABlock.forward_combined_with_keyframes`.
        ``mul_`` rather than ``*`` keeps the buffer identity the in-place chunked pathway
        depends on, so the next block's deferred inject writes into the same tensor.
        """
        x, keyframe_x = residual_attn_with_keyframes(
            x,
            keyframe_x,
            self.attn,
            self.norm1,
            scale_msa,
            shift_msa,
            keyframe_planes,
        )
        x = residual_modulating_mlp(x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        keyframe_x = residual_modulating_mlp(keyframe_x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        keyframe_x.mul_(keyframe_planes.valid.device_tensor[None, :, None, None, None])
        return x, keyframe_x

    def forward_x_ctx_with_keyframes(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        keyframe_x: torch.Tensor,
        keyframe_stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        keyframe_planes: KeyframePlanes,
        *,
        drop_leading_frame: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deferred-context dual-stream block: per-stream inject, then joint attn + MLP.
        The keyframe inject always collapses its temporal stride (each plane is a ``T=1``
        clip), so ``drop_leading_frame`` applies to the video stream only -- it is a tiling
        property and must not leak into the keyframe arm.
        """
        if self.stage4_upsample is None:
            raise RuntimeError("stage4_upsample not configured; apply DiffVAEConfig with block=CHUNKED")

        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        upsample = self.stage4_upsample
        stride: tuple[int, int, int] = tuple(upsample.stride)  # type: ignore[assignment]
        x = inject_context(
            x,
            stage4_feat,
            upsample.proj,
            self.context_proj,
            stride,
            w_chunks=self.attn.w_chunks,
            drop_leading_frame=drop_leading_frame,
        )
        keyframe_x = inject_keyframe_context(
            keyframe_x,
            keyframe_stage4_feat,
            upsample.proj,
            self.context_proj,
            stride,
            w_chunks=self.attn.w_chunks,
        )
        return self.forward_attn_mlp_with_keyframes(
            x,
            keyframe_x,
            scale_msa,
            shift_msa,
            scale_mlp,
            shift_mlp,
            keyframe_planes,
        )

    def forward_x_ctx(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        if self.stage4_upsample is None:
            raise RuntimeError("stage4_upsample not configured; apply DiffVAEConfig with block=CHUNKED")

        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        upsample = self.stage4_upsample
        x = inject_context(
            x,
            stage4_feat,
            upsample.proj,
            self.context_proj,
            tuple(upsample.stride),  # type: ignore[arg-type]
            w_chunks=self.attn.w_chunks,
            drop_leading_frame=drop_leading_frame,
        )
        return self.forward_attn_mlp(x, scale_msa, shift_msa, scale_mlp, shift_mlp)

    def forward(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        return self.forward_x_ctx(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame)
