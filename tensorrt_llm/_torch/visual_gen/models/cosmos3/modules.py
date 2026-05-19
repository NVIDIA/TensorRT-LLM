import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.utils import spectral_norm, weight_norm

# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version
        based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features

        self.alpha_logscale = alpha_logscale
        param_shape = (1, in_features, 1)
        if self.alpha_logscale:
            self.alpha = Parameter(torch.zeros(param_shape) * alpha)
            self.beta = Parameter(torch.zeros(param_shape) * alpha)
        else:
            self.alpha = Parameter(torch.ones(param_shape) * alpha)
            self.beta = Parameter(torch.ones(param_shape) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep compatibility with checkpoints storing Snake params as either [C] or [1, C, 1].
        alpha = self.alpha if self.alpha.ndim == 3 else self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta if self.beta.ndim == 3 else self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)

        return x + (1.0 / (beta + 1e-9)) * pow(torch.sin(x * alpha), 2)


# ---------------------------------------------------------------------------
# WN wrappers
# ---------------------------------------------------------------------------


def WNConv1d(*args: Any, **kwargs: Any) -> nn.Conv1d:
    """Weight-normalized 1D convolution."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args: Any, **kwargs: Any) -> nn.ConvTranspose1d:
    """Weight-normalized 1D transpose convolution."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# ---------------------------------------------------------------------------
# EnCodec-style conv helpers (SConv1d / SConvTranspose1d)
# ---------------------------------------------------------------------------

CONV_NORMALIZATIONS = frozenset(
    ["none", "weight_norm", "spectral_norm", "time_layer_norm", "layer_norm", "time_group_norm"]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
) -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.LayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    return nn.Identity()


def pad1d(x: torch.Tensor, paddings: tuple, mode: str = "zero", value: float = 0.0) -> torch.Tensor:
    """Tiny wrapper around F.pad that handles reflect padding on short inputs."""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tuple) -> torch.Tensor:
    """Remove padding from x. Only for 1D."""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConvTranspose1d(nn.Module):
    """ConvTranspose1d with optional weight_norm / spectral_norm."""

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with builtin asymmetric/causal padding and normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1.0, (
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        )
        assert 0.0 <= self.trim_right_ratio <= 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y
