"""Stage-5 block chain ping-pong buffers for fused DSL launches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae import decode_workspace
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.layers import ChannelLinear

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.keyframes import KeyframePlanes


def alloc_block_volume(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """A ``(1,T,H,W,C)`` tensor whose storage carries the fused kernel's lane slack, leased from the active
    decode workspace until its tile ends (``torch.empty`` style without one)."""
    _, t, h, w, c = shape
    rows = decode_workspace.volume((t * h * w + decode_workspace.out_slack_rows(), c), device=device, dtype=dtype)
    return rows[: t * h * w].view(1, t, h, w, c)


def linear_into_block_volume(linear: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """``linear(x)`` written straight into an :func:`alloc_block_volume` buffer."""
    out = alloc_block_volume((*x.shape[:-1], linear.out_features), device=x.device, dtype=x.dtype)
    n_rows = out.numel() // linear.out_features
    rows = out.view(n_rows, linear.out_features)
    torch.mm(x.reshape(n_rows, linear.in_features), linear.weight.t(), out=rows)
    if linear.bias is not None:
        rows.add_(linear.bias)
    return out


def _recyclable_rows(volume: torch.Tensor) -> torch.Tensor | None:
    """``volume``'s backing ``(rows, C)`` buffer, if it was allocated with lane slack."""
    if volume.ndim != 5 or volume.dtype != torch.bfloat16 or volume.storage_offset() != 0:
        return None
    channels = volume.shape[-1]
    n_rows = volume.numel() // channels
    want = n_rows + decode_workspace.out_slack_rows()
    have = volume.untyped_storage().nbytes() // (volume.element_size() * channels)
    if not volume.is_contiguous() or have < want:
        return None
    return torch.as_strided(volume, (want, channels), (channels, 1))


class DSLChannelLinear(ChannelLinear):
    """``conv_in_x_t`` producing a volume the fused block chain can recycle."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_into_block_volume(self, x)


class DSLDiffusionBlockChain(nn.ModuleList):
    """Stage-5 blocks as one callable chain, alternating two output buffers per stream.
    Signature matches deferred decode: ``(x, stage4_feat, modulation)``, with
    :meth:`forward_x_ctx_with_keyframes` as the joint entry.
    A block cannot write into its own input -- it reads neighborhoods of it -- so the
    chain keeps the buffer it *stopped* reading and hands that back as the next block's
    output. Two volumes per stream for the whole stage, rather than one per block. The
    first block still allocates, because the entry volume is the caller's and the chain
    has nothing spare yet; from the second on it alternates.
    """

    def forward(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        spare: torch.Tensor | None = None
        for block in self:
            y = block(
                x,
                stage4_feat,
                modulation,
                drop_leading_frame=drop_leading_frame,
                out=spare,
            )
            spare = _recyclable_rows(x)
            x = y
        return x

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
        """The joint chain, recycling both streams' buffers.
        The anchor stream alternates on its own pair: its volume is ``n_kf`` planes against
        the video's ``T`` frames, so the two buffer sizes differ and one pool cannot serve
        both. Same rule per stream -- last block's input becomes this block's output. A
        block that drops invalid planes returns a tight anchor tensor, which simply offers
        nothing to recycle on the next hop.
        """
        spare: torch.Tensor | None = None
        keyframe_spare: torch.Tensor | None = None
        for block in self:
            y, keyframe_y = block.forward_x_ctx_with_keyframes(
                x,
                stage4_feat,
                keyframe_x,
                keyframe_stage4_feat,
                modulation,
                keyframe_planes,
                drop_leading_frame=drop_leading_frame,
                out=spare,
                out_keyframes=keyframe_spare,
            )
            spare, keyframe_spare = _recyclable_rows(x), _recyclable_rows(keyframe_x)
            x, keyframe_x = y, keyframe_y
        return x, keyframe_x
