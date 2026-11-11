from typing import Set, Tuple

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.attention import AttentionType, make_attn
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.causality_axis import CausalityAxis
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.resnet import ResnetBlock
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.common.normalization import NormType


class Downsample(torch.nn.Module):
    """
    A downsampling layer that can use either a strided convolution
    or average pooling. Supports standard and causal padding for the
    convolutional mode.
    """

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and not self.with_conv:
            raise ValueError("causality is only supported when `with_conv=True`.")

        if self.with_conv:
            # Do time downsampling here
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # Padding tuple is in the order: (left, right, top, bottom).
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pad = (0, 1, 0, 1)
                case CausalityAxis.WIDTH:
                    pad = (2, 0, 0, 1)
                case CausalityAxis.HEIGHT:
                    pad = (0, 1, 2, 0)
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pad = (1, 0, 0, 1)
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # This branch is only taken if with_conv=False, which implies causality_axis is NONE.
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        return x


def build_downsampling_path(  # noqa: PLR0913
    *,
    ch: int,
    ch_mult: Tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    attn_resolutions: Set[int],
    resamp_with_conv: bool,
) -> tuple[torch.nn.ModuleList, int]:
    """Build the downsampling path with residual blocks, attention, and downsampling layers."""
    down_modules = torch.nn.ModuleList()
    curr_res = resolution
    in_ch_mult = (1, *tuple(ch_mult))
    block_in = ch

    for i_level in range(num_resolutions):
        block = torch.nn.ModuleList()
        attn = torch.nn.ModuleList()
        block_in = ch * in_ch_mult[i_level]
        block_out = ch * ch_mult[i_level]

        for _ in range(num_res_blocks):
            block.append(
                ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    causality_axis=causality_axis,
                )
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                attn.append(make_attn(block_in, attn_type=attn_type, norm_type=norm_type))

        down = torch.nn.Module()
        down.block = block
        down.attn = attn
        if i_level != num_resolutions - 1:
            down.downsample = Downsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res = curr_res // 2
        down_modules.append(down)

    return down_modules, block_in
