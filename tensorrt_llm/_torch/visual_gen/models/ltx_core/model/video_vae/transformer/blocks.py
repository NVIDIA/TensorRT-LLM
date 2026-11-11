"""NABlock and DiffusionNABlock parameter shells for DiffVAE.
Pathway subclasses live in ``chunked/`` and ``combined/``; ``apply`` installs
them via ``__class__`` swap (same pattern as ``Fp8CastLinear``). The shell owns
weights + shared AdaLN helpers only — no pathway forward.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae import decode_workspace
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.det_rms_norm import DetRMSNorm
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.layers import AdaLNZero
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.swiglu import SwiGLU, plain_mlp, plain_mlp_accumulate

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import KeyframeStream

__all__ = [
    "DiffusionNABlock",
    "NABlock",
]


class NABlock(nn.Module):
    """Pre-norm transformer block: NA -> SwiGLU MLP with residual adds."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = DetRMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = DetRMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Channels-last in/out: (B, T, H, W, C). The norms pick their kernel themselves (:class:`DetRMSNorm`)."""
        x = x + self.attn(self.norm1(x))
        return plain_mlp(x, self.mlp, self.norm2, self.mlp.tile)

    def forward_with_keyframes(
        self,
        x: torch.Tensor,
        keyframes: KeyframeStream,
    ) -> tuple[torch.Tensor, KeyframeStream]:
        """Dual-stream block: video ``(B,T,H,W,C)`` and keyframe planes ``(B,P,H,W,C)``.
        Every weight is shared with :meth:`forward`; the streams meet only inside the
        joint attention softmax. Invalid planes are deliberately *not* re-zeroed here --
        the decoder re-zeroes after each upsample instead, matching upstream, so a masked
        plane's hidden state may drift within a stage. It is masked out of every softmax
        regardless, so this is cosmetic; reproducing it keeps us comparable.
        Eager calls carve the video stream's per-block temporaries from the decode workspace's
        idle volumes (none outlives the block, so the arena is reset at both block boundaries)
        and add the residuals into the fresh attention and MLP outputs; same values.
        """
        if torch.compiler.is_compiling():
            attn_out, keyframe_attn = self.attn.forward_with_keyframes(
                self.norm1(x), dataclasses.replace(keyframes, x=self.norm1(keyframes.x))
            )
            x = x + attn_out
            keyframe_x = keyframes.x + keyframe_attn.x
            x = plain_mlp(x, self.mlp, self.norm2, self.mlp.tile)
            keyframe_x = plain_mlp(keyframe_x, self.mlp, self.norm2, self.mlp.tile)
            return x, dataclasses.replace(keyframes, x=keyframe_x)
        decode_workspace.phase_reset()
        attn_out, keyframe_attn = self.attn.forward_with_keyframes(
            self.norm1(x), dataclasses.replace(keyframes, x=self.norm1(keyframes.x))
        )
        x = attn_out.add_(x)
        keyframe_x = keyframe_attn.x.add_(keyframes.x)
        x = plain_mlp_accumulate(x, self.mlp, self.norm2, self.mlp.tile)
        keyframe_x = plain_mlp_accumulate(keyframe_x, self.mlp, self.norm2, self.mlp.tile)
        decode_workspace.phase_reset()
        return x, dataclasses.replace(keyframes, x=keyframe_x)


class DiffusionNABlock(nn.Module):
    """Parameter shell for diffusion NA + SwiGLU with shared AdaLN-Zero.
    Mode-specific subclasses (:class:`~ltx_core.model.video_vae.transformer.combined.block.CombinedDiffusionNABlock`,
    :class:`~ltx_core.model.video_vae.transformer.chunked.block.ChunkedDiffusionNABlock`)
    are installed via ModuleOps ``__class__`` swap and own the forward path.
    Not a ``Protocol``: must be a concrete ``nn.Module`` so checkpoint load and
    ``__class__`` swap keep one parameter identity.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        rope_dim_split: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.context_channels = context_channels
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))

        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim, rope_dim_split=rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)
        self.attn.proj.reset_parameters()

    def _modulation(
        self, modulation: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        return scale_msa, shift_msa, scale_mlp, shift_mlp
