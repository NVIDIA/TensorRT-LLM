from typing import Set, Tuple

import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.attention import AttentionType, make_attn
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.causal_conv_2d import make_conv2d
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.causality_axis import CausalityAxis
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.resnet import ResnetBlock
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.common.normalization import NormType


class Upsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(in_channels, in_channels, kernel_size=3, stride=1, causality_axis=causality_axis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            # Drop FIRST element in the causal axis to undo encoder's padding, while keeping the length 1 + 2 * n.
            # For example, if the input is [0, 1, 2], after interpolation, the output is [0, 0, 1, 1, 2, 2].
            # The causal convolution will pad the first element as [-, -, 0, 0, 1, 1, 2, 2],
            # So the output elements rely on the following windows:
            # 0: [-,-,0]
            # 1: [-,0,0]
            # 2: [0,0,1]
            # 3: [0,1,1]
            # 4: [1,1,2]
            # 5: [1,2,2]
            # Notice that the first and second elements in the output rely only on the first element in the input,
            # while all other elements rely on two elements in the input.
            # So we can drop the first element to undo the padding (rather than the last element).
            # This is a no-op for non-causal convolutions.
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pass  # x remains unchanged
                case CausalityAxis.HEIGHT:
                    x = x[:, :, 1:, :]
                case CausalityAxis.WIDTH:
                    x = x[:, :, :, 1:]
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pass  # x remains unchanged
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        return x


def build_upsampling_path(  # noqa: PLR0913
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
    initial_block_channels: int,
) -> tuple[torch.nn.ModuleList, int]:
    """Build the upsampling path with residual blocks, attention, and upsampling layers."""
    up_modules = torch.nn.ModuleList()
    block_in = initial_block_channels
    curr_res = resolution // (2 ** (num_resolutions - 1))

    for level in reversed(range(num_resolutions)):
        stage = torch.nn.Module()
        stage.block = torch.nn.ModuleList()
        stage.attn = torch.nn.ModuleList()
        block_out = ch * ch_mult[level]

        for _ in range(num_res_blocks + 1):
            stage.block.append(
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
                stage.attn.append(make_attn(block_in, attn_type=attn_type, norm_type=norm_type))

        if level != 0:
            stage.upsample = Upsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res *= 2

        up_modules.insert(0, stage)

    return up_modules, block_in
