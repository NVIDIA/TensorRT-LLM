import torch
import torch.nn.functional as F

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx_core.model.audio_vae.causality_axis import CausalityAxis


class CausalConv2d(torch.nn.Module):
    """
    A causal 2D convolution.
    This layer ensures that the output at time `t` only depends on inputs
    at time `t` and earlier. It achieves this by applying asymmetric padding
    to the time dimension (width) before the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()

        self.causality_axis = causality_axis

        # Ensure kernel_size and dilation are tuples
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        dilation = torch.nn.modules.utils._pair(dilation)

        # Calculate padding dimensions
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        # The padding tuple for F.pad is (pad_left, pad_right, pad_top, pad_bottom)
        match self.causality_axis:
            case CausalityAxis.NONE:
                self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            case CausalityAxis.WIDTH | CausalityAxis.WIDTH_COMPATIBILITY:
                self.padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
            case CausalityAxis.HEIGHT:
                self.padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
            case _:
                raise ValueError(f"Invalid causality_axis: {causality_axis}")

        # The internal convolution layer uses no padding, as we handle it manually
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply causal padding before convolution
        x = F.pad(x, self.padding)
        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int = 1,
    padding: tuple[int, int, int, int] | None = None,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis | None = None,
) -> torch.nn.Module:
    """
    Create a 2D convolution layer that can be either causal or non-causal.
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        stride: Convolution stride
        padding: Padding (if None, will be calculated based on causal flag)
        dilation: Dilation rate
        groups: Number of groups for grouped convolution
        bias: Whether to use bias
        causality_axis: Dimension along which to apply causality.
    Returns:
        Either a regular Conv2d or CausalConv2d layer
    """
    if causality_axis is not None:
        # For causal convolution, padding is handled internally by CausalConv2d
        return CausalConv2d(in_channels, out_channels, kernel_size, stride, dilation, groups, bias, causality_axis)
    else:
        # For non-causal convolution, use symmetric padding if not specified
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else tuple(k // 2 for k in kernel_size)

        return torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
